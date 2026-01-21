#ifndef OPS_SIGMOID_H
#define OPS_SIGMOID_H

#include "../operator.h"
#include <cmath>      
#include <algorithm>  
#include <limits>     

class SigmoidOperator : public Operator
{
public:
    void compute([[maybe_unused]]const Node& node, const std::vector<Tensor<float>*>& inputs, Tensor<float>& output) override
    {
        const Tensor<float>* input {inputs[0]};
        output = Tensor<float>(input->shape()); // set output shape to same shape as input

        const float* in_ptr {input->data()};
        float* out_ptr {output.data()};
        std::size_t total_elements {input->size()};

        // apply sigmoid func to tensor elem-wise
        for (std::size_t i{}; i < total_elements; ++i) 
        {
            float val {in_ptr[i]};
            out_ptr[i] = 1 / (1 + std::exp(-val));   // Sigmoid: f(x) = 1 / (1 + exp(-x))
        }
    }
};

#endif
