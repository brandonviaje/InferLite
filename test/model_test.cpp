#include <iostream>
#include <fstream>
#include <cassert>
#include "../src/onnx-ml.pb.h"

void test_protobuf_instantiation()
{
    onnx::ModelProto model;
    model.set_ir_version(8);
    assert(model.ir_version() == 8);
    std::cout << "ONNX ModelProto instantiated and checked!" << std::endl;
}

int main()
{
    try
    {
        test_protobuf_instantiation();
        std::cout << "ALL MODEL TESTS PASSED!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Model test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}