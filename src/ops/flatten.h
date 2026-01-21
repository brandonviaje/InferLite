#ifndef OPS_FLATTEN_H
#define OPS_FLATTEN_H

#include "../operator.h"

class FlattenOperator : public Operator 
{
public:
    void compute([[maybe_unused]]const Node& node, const std::vector<Tensor<float>*>& inputs, Tensor<float>& output) override 
    {
        
        // get input
        const Tensor<float>* input = inputs[0];

        // calculate new shape, flatten keeps the batch size (dim 0) and collapse the rest.
        // [N, C, H, W] -> [N, C*H*W]
        std::size_t batch {input->shape()[0]}; 
        std::size_t total_elements {input->size()};
        std::size_t feature_size {total_elements / batch};

        output = *input;                        // copy data
        output.reshape({batch, feature_size});  // update shape [N, C, H, W] -> [N, C*H*W]
    }
};

#endif
