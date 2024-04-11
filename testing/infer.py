import tensorrt as trt
from tensorrt import TensorIOMode
import numpy as np
import os
import time
from PIL import Image
import cv2
import json

import pycuda.driver as cuda
import pycuda.autoinit

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path, max_batch_size=1, dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

        print("max batch size:: {}".format(self.max_batch_size))

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
        
        np.copyto(self.inputs[0].host, x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        self.stream.synchronize()
        return [out.host.reshape(batch_size, -1) for out in self.outputs]

def load_annotation(path):
    with open('test.json') as f:
        d = json.load(f)
    return d

def resize_image(imgage, shape):
    w, h = shape[2], shape[3]
    resized_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return np.array(resized_image)

if __name__ == "__main__":
    img_path = "/home/ubuntu/workstation/Anchor3DLane/data/OpenLane/images/training/segment-268278198029493143_1400_000_1420_000_with_camera_labels/155815122108723600.jpg"
    image = cv2.imread(img_path)
    
    # batch_size = 1
    # trt_engine_path = os.path.join("test.plan")
    # model = TrtModel(trt_engine_path)
    
    image_array = []

    batch_size = 1
    trt_engine_path = os.path.join("engines/lane_single_batch.plan")
    model = TrtModel(trt_engine_path)
    
    shape = model.engine.get_tensor_shape("input")
    output_shape = model.engine.get_tensor_shape("output")
    print("output shape of the network:: {}".format(output_shape))

    start = time.time()

    resized_img = resize_image(image, shape).reshape((shape[1], shape[2], shape[3]))

    for i in range(batch_size):
        image_array.append(resized_img)
    
    image_array = np.array(image_array) / 255.

    # data = np.random.randint(0, 255,(16, 3, 360, 480))/255
    # print(image_array.shape)
    # data = np.random.randint(0, 255, (batch_size, *shape[1:]))/255

    result = model(image_array, batch_size)
    
    # seg_pred shape -> 1, 4431, 86
    seg_pred = result[0].reshape((batch_size, output_shape[1], output_shape[2]))
    print(seg_pred)
    
    end = time.time()
    
    print("*" * 100)
    print(f"The execution took {(end-start)}.")
    print("*" * 100)
