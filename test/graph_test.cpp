#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../src/graph.h"
#include "../src/tensor.h"

// helper to check if two floats are close
bool is_close(float a, float b, float epsilon = 1e-4) 
{
    return std::abs(a - b) < epsilon;
}

void test_manual_relu_graph() 
{
    std::cout << "Manual Graph Construction (Relu)..." << std::endl;
    Graph graph;                 // build computational graph                    

    // input x
    auto* input_info {graph.graph_proto.add_input()};
    input_info->set_name("X");      
    
    // output y
    auto* output_info {graph.graph_proto.add_output()};
    output_info->set_name("Y");

    // create node
    auto* node {graph.graph_proto.add_node()};
    node->set_op_type("Relu");   // match registry string
    node->set_name("Relu_1");
    node->add_input("X");        // connect to input
    node->add_output("Y");       // connect to output

    // run inference
    
    // create input tensor, ReLu should change each node
    Tensor<float> input({1, 4});
    input.data()[0] = -10.0f;    // should become 0 
    input.data()[1] = 0.0f;      // should stay 0
    input.data()[2] = 5.5f;      // should stay 5.5
    input.data()[3] = -0.1f;     // should become 0

    Tensor<float> output;
    
    try 
    {
        graph.infer(input, output);
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        exit(1);
    }

    // assertions
    const float* out_ptr = output.data();
    
    // should become 0: -10.0 -> 0.0
    assert(is_close(out_ptr[0], 0.0f)); 
    std::cout << "  [-10.0 -> " << out_ptr[0] << "] OK" << std::endl;

    // should stay 0: 0.0 -> 0.0
    assert(is_close(out_ptr[1], 0.0f));
    std::cout << "  [0.0 -> " << out_ptr[1] << "] OK" << std::endl;

    // should stay 5.5: 5.5 -> 5.5
    assert(is_close(out_ptr[2], 5.5f));
    std::cout << "  [5.5 -> " << out_ptr[2] << "] OK" << std::endl;

    // should become 0: -0.1 -> 0.0
    assert(is_close(out_ptr[3], 0.0f));
    std::cout << "  [-0.1 -> " << out_ptr[3] << "] OK" << std::endl;

    std::cout << "[PASSED] Relu graph logic passed \n" << std::endl;
}

int main() 
{
    test_manual_relu_graph();
    std::cout << "GRAPH TESTS PASSED!" << std::endl;
    return 0;
}
