#ifndef OPS_RELU_H
#define OPS_RELU_H

#include "../operator.h"

class ReluOperator : public Operator
{
public:
    void forward(const std::vector<Tensor<float>*>& inputs, std::vector<Tensor<float>*>& outputs) override
    {
        const Tensor<float>* input = inputs[0]; // input tensor
        Tensor<float>* output = outputs[0];     // get existing output tensor 
        output->resize(input->shape());         // resize tensor to match input

        // pointers
        const float* in_data = input->data();
        float* out_data = output->data();
        std::size_t size = input->size();

        // apply ReLU element wise
        for (std::size_t i = 0; i < size; ++i) 
        {
            float val = in_data[i];
            out_data[i] = (val > 0.0f) ? val : 0.0f; // f(x) = max(0,x)
        }
    }
};

#endif
