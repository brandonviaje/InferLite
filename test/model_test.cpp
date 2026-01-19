#include <iostream>
#include <fstream>
#include <cassert>
#include "../src/onnx-ml.pb.h"
#include "../src/loader.h"
#include "../src/tensor.h"

void test_protobuf_instantiation()
{
    onnx::ModelProto model;
    model.set_ir_version(8);
    assert(model.ir_version() == 8);
    std::cout << "ONNX ModelProto instantiated and checked!" << std::endl;
}

void test_load_onnx_tensor()
{
    // ceate dummy ONNX TensorProto
    onnx::TensorProto proto;
    proto.add_dims(2);
    proto.add_dims(2);
    proto.set_data_type(onnx::TensorProto::FLOAT);

    std::vector<float> original_data = {1.1f, 2.2f, 3.3f, 4.4f};

    // copy floats into raw byte string
    proto.set_raw_data(std::string(reinterpret_cast<const char *>(original_data.data()), original_data.size() * sizeof(float)));

    // use Loader
    Tensor<float> tensor = Loader::load_tensor(proto);

    // assertions
    assert(tensor.shape()[0] == 2);
    assert(tensor.shape()[1] == 2);
    assert(tensor.at({0, 0}) == 1.1f);
    assert(tensor.at({1, 1}) == 4.4f);

    std::cout << "ONNX Tensor conversion passed!" << std::endl;
}

int main()
{
    try
    {
        test_protobuf_instantiation();
        test_load_onnx_tensor();
        std::cout << "MODEL TESTS PASSED!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Model test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}