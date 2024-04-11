import torch
import tensorrt as trt


def build_engine(onnx_model_path: str, trt_engine_path: str, trt_logger_enable=True):
    """
    Builds and then writes a serialized tensorrt engine. (Can then be desrialized in C++_
    :param onnx_model_path: path to the onnx model
    :param trt_engine_path: path to store the TRT engine
    :param trt_logger_enable: flag to enable/disable trt logging
    """
    
    TRT_LOGGER = trt.Logger()
    builder = trt.Builder(TRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, TRT_LOGGER)
    success = parser.parse_from_file(onnx_model_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    config = builder.create_builder_config()
    # config.max_workspace_size = 1 << 22 # 1 MiB

    serialized_engine = builder.build_serialized_network(network, config)
    with open(trt_engine_path, "wb") as f:
        f.write(serialized_engine)
    

if __name__ == "__main__":
    onnx_path = "engines/test.onnx"
    trt_path = "engines/test.plan"
    build_engine(onnx_path, trt_path, False)