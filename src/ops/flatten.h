#ifndef OPS_FLATTEN_H
#define OPS_FLATTEN_H

#include "../operator.h"

class FlattenOperator : public Operator 
{
public:
    void set_attributes(const Node& node) override 
    {
        // ONNX Default is axis=1 
        axis_ = node.get_attribute<int64_t>("axis").value_or(1);
    }
    
    void forward(const std::vector<Tensor<float>*>& inputs, std::vector<Tensor<float>*>& outputs) override
    {
        const Tensor<float>* input = inputs[0];
        Tensor<float>* output = outputs[0];
        const std::vector<std::size_t>& shape = input->shape();

        int axis = static_cast<int>(axis_);
        if (axis < 0) axis += shape.size();

        std::size_t batch = 1;
        for (int i = 0; i < axis; ++i) batch *= shape[i];

        std::size_t features = 1;
        for (size_t i = axis; i < shape.size(); ++i) features *= shape[i];

        output->resize({batch, features});
        std::copy(input->data(), input->data() + input->size(), output->data());
    }

private:
    int64_t axis_ = 1;
};

#endif
