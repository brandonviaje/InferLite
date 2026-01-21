#ifndef GRAPH_H
#define GRAPH_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

#include "tensor.h"
#include "node.h"
#include "registry.h"
#include "onnx-ml.pb.h" 

class Graph 
{
public:
    onnx::GraphProto graph_proto;
    std::unordered_map<std::string, Tensor<float>> weights;
    void load(const std::string& filepath);                          // load onnx file + weights
    void infer(const Tensor<float>& input, Tensor<float>& output);   // run inference
private:
    void load_weights(const onnx::GraphProto& graph);
};

#endif
