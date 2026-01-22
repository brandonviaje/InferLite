#ifndef OPERATOR_H
#define OPERATOR_H

#include <vector>
#include <string>
#include "tensor.h"
#include "node.h"

class Operator 
{
public:
    virtual ~Operator() = default;                                                                                // virtual destructor
    virtual void set_attributes(const Node& node) { (void)node; }                                                 // load settings
    virtual void forward(const std::vector<Tensor<float>*>& inputs, std::vector<Tensor<float>*>& outputs) = 0; // execute operator
};

#endif
