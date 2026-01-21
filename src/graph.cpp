#include "graph.h"
#include "registry.h"
#include "node.h"
#include <fstream>
#include <iostream>
#include <cstring> 

// load onnx file + weights
void Graph::load(const std::string& filepath)
{
    std::ifstream input(filepath, std::ios::ate | std::ios::binary); 

    if (!input.is_open())
    {
        throw std::runtime_error("failed to open: " + filepath);
    }

    std::size_t size = static_cast<std::size_t>(input.tellg());
    input.seekg(0, std::ios::beg);

    // read file into buffer
    std::string buffer(size, ' ');
    input.read(&buffer[0], size);

    // parse proto buffer
    onnx::ModelProto model_proto;

    if (!model_proto.ParseFromString(buffer)) 
    {
        throw std::runtime_error("failed to parse onnx");
    }

    graph_proto = model_proto.graph();

    // load weights
    load_weights(graph_proto);

    std::cout << "Graph loaded: " << weights.size() << " weight tensors." << std::endl;
}
 
// run inference
void Graph::infer(const Tensor<float>& input, Tensor<float>& output) {

    // init workspace
    std::unordered_map<std::string, Tensor<float>> activations;
    
    // get tensor from workspace, fall back to weights
    auto get_tensor = [&](const std::string& name) -> Tensor<float>* 
    {
        if (activations.count(name)) return &activations.at(name);  // check if tensor was produced earlier from inference
        if (weights.count(name)) return &weights.at(name);          // else check if tensor is a constant model param
        throw std::runtime_error("tensor not found: " + name);
    };

    // copy input into workspace
    std::string input_name {graph_proto.input(0).name()};
    activations[input_name] = input;

    // execute nodes sequentially
    for (const auto& node_proto : graph_proto.node()) 
    {
        // create operator
        auto op = Registry::create(node_proto.op_type());
        if (!op) continue; // skip unsupported operators
 
        // get inputs
        std::vector<Tensor<float>*> op_inputs;

        for (const auto& in_name : node_proto.input()) 
        {
            op_inputs.push_back(get_tensor(in_name));
        }

        // compute
        Node node_wrapper(node_proto); 
        Tensor<float> op_output;
        op->compute(node_wrapper, op_inputs, op_output);
        
        // save result
        std::string out_name {node_proto.output(0)};
        activations[out_name] = std::move(op_output);
    }

    // get final output
    std::string final_out_name {graph_proto.output(0).name()};
    output = std::move(activations.at(final_out_name));
}

// helper to load weights
void Graph::load_weights(const onnx::GraphProto& graph) 
{ 
    // iterate through graph
    for (const auto& initializer : graph.initializer()) 
    {
        std::vector<size_t> dims;
        for (auto d : initializer.dims()) dims.push_back(d);

        Tensor<float> tensor(dims);

        // handle raw binary
        if (initializer.has_raw_data()) 
        {
            const std::string& raw {initializer.raw_data()};
            std::memcpy(tensor.data(), raw.data(), raw.size());
        } 
        else  // handle float list
        {
            for (int i = 0; i < initializer.float_data_size(); ++i) 
            {
                tensor.data()[i] = initializer.float_data(i);
            }
        }
        weights[initializer.name()] = std::move(tensor);
    }
}
