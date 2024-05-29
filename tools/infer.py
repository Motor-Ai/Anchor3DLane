import tensorrt as trt
import numpy as np
import os
import time
from PIL import Image
import cv2
import torch
import json
import torch.nn as nn
import postprocess
import ctypes

import pycuda.driver as cuda
import pycuda.autoinit

@torch.no_grad()
def nms_3d(proposals, scores,  thresh, anchor_len=10):
    
    order = scores.argsort(descending=True)
    keep = []

    while order.shape[0] > 0:
    
        i = order[0]
        keep.append(i)
        x1 = proposals[i][5:5+anchor_len]  # [l]
        z1 = proposals[i][5+anchor_len:5+anchor_len*2]   # [l]

        x2s = proposals[order[1:]][:, 5:5+anchor_len]  # [n, l]
        z2s = proposals[order[1:]][:, 5+anchor_len:5+anchor_len*2]   # [n, l]
        torch.set_printoptions(sci_mode=False)

        dis = ((x1 - x2s) ** 2 + (z1 - z2s) ** 2) ** 0.5  # [n, l]
        dis = (dis).sum(dim=1) / (20)  # [n], incase no matched points
        inds = torch.where(dis > thresh)[0]  # [n']
        order = order[inds + 1]   # [n']
    return torch.tensor(keep)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        assert os.path.exists(engine_path)
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            elif self.engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
       
    def __call__(self,x:np.ndarray, batch_size=1):
        
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        self.stream.synchronize()
        return [out.host.reshape(batch_size, -1) for out in self.outputs]

def load_annotation(path):
    with open(path) as f:
        d = json.load(f)
    return d

def resize_image(img_arr, shape):
    w, h = shape[2], shape[3]
    resized_img = cv2.resize(img_arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.transpose(resized_img)

def convert3d_to_2dimage(anno):
    en = np.vstack(anno['extrinsic'])
    in_ = np.vstack(anno['intrinsic'])
    xyz = np.vstack(anno['lane_lines'][0]['xyz'])
    ones = np.ones((1, len(xyz[0])))
    point_3d_homo = np.concatenate((xyz, ones), axis=0)
    projection_matrix = np.dot(in_, en[:3])
    point_2d_homogeneous = np.dot(projection_matrix, point_3d_homo)
    point_2d = (point_2d_homogeneous[:2] / point_2d_homogeneous[2])

    return point_2d

def load_querry_image_list(path):
    if os.path.isfile(path): 
        return [path]
    elif os.path.isdir(path):
        files = os.listdir(path)
        return [os.path.join(path, i) for i in files]

@torch.jit.script
def custom_cumsum(input_, axis: int):
    # Initialize the output tensor with zeros
    output = torch.zeros_like(input_)
    if axis == 1:
        # Iterate over each row in the input tensor
        for i in range(input_.size(0)):
            # Iterate over each element in the row
            for j in range(input_.size(1)):
                if j == 0:
                    # If it's the first element in the row, just copy it to the output
                    output[i][j] = input_[i][j]
                else:
                    # Otherwise, accumulate the sum of previous elements in the row
                    output[i][j] = output[i][j - 1] + input_[i][j]
    else:
        raise ValueError("Axis value other than 1 is not supported")
    return output


def python_nms(proposals, nms_thres=0, conf_threshold=None, refine_vis=False, vis_thresh=0.5):
    softmax = nn.Softmax(dim=1)
    
    scores = 1 - softmax(proposals[:, 5 + 20 * 3:5 + 20 * 3+21])[:, 0]  # pos_score  # for debug
    anchor_inds = torch.arange(proposals.shape[0], device=proposals.device)
    if conf_threshold > 0:
        above_threshold = scores > conf_threshold
        proposals = proposals[above_threshold]
        scores = scores[above_threshold]
        anchor_inds = anchor_inds[above_threshold]
    
    if nms_thres > 0:
        keep = nms_3d(proposals, scores, thresh=nms_thres, anchor_len=20)
        proposals = proposals[keep]
        anchor_inds = anchor_inds[keep]
    return proposals


if __name__ == "__main__":
    querry_path = "/home/sandhu/project/LaneSeg/lane_prod/test_images/"
    anno_path = "test_anno.json"
    trt_engine_path = os.path.join("test.plan")
    images_path = load_querry_image_list(querry_path)
    batch_size = 1
    images_path = ["test_images/150912151202610600.jpg"]
    model = TrtModel(trt_engine_path)
    shape = model.engine.get_tensor_shape("input")

    torch.set_printoptions(sci_mode=False)

    for i in images_path:
        image = cv2.imread(i)
        resized_img = resize_image(image, shape)
        result = model(resized_img, batch_size)
        proposals = torch.tensor(result[0].reshape(1,-1, 86))

        # Python output
        python_out = python_nms(proposals[0], 2, 0.2, True, 0.5)
        python_out = python_out.numpy().astype(np.float32).flatten()
        np.savetxt("ooutput.txt", python_out, fmt='%1.4f')

        # print(len(python_out))

        # Proposals flattened for cpp
        # proposals = proposals.numpy().astype(np.float32).flatten()
        # np.savetxt("proposals.txt", proposals, fmt='%1.4f')
        # np.printoptions(suppress=True)
        # cpp_out = postprocess.nms(proposals, 2, 0.2, False, 0.5)
        