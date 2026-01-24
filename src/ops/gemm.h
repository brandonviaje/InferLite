#ifndef OPS_GEMM_H
#define OPS_GEMM_H

#include "../operator.h"
#include "../attribute.h"
#include "../tensor.h"
#include <cmath>

class GemmOperator : public Operator
{
public:
    void set_attributes(const Node& node) override 
    {
        alpha_  = node.get_attribute<float>("alpha").value_or(1.0f);
        beta_   = node.get_attribute<float>("beta").value_or(1.0f);
        transA_ = node.get_attribute<int64_t>("transA").value_or(0);
        transB_ = node.get_attribute<int64_t>("transB").value_or(0);
    }

    void forward(const std::vector<Tensor<float>*>& inputs, std::vector<Tensor<float>*>& outputs) override
    {
        const auto* A = inputs[0];
        const auto* B = inputs[1];
        
        std::size_t M = A->rows();
        std::size_t K = A->cols();
        
        // handle transposeA (swap M and K if transA is on)
        if (transA_) 
        {
            M = A->cols();
            K = A->rows();
        }

        std::size_t N = (B->cols() == K) ? B->rows() : B->cols(); 
        bool transpose_B = (B->cols() == K); 

        // prepare output
        outputs[0]->resize({M, N});
        
        float* Y = outputs[0]->data();
        const float* A_data = A->data();
        const float* B_data = B->data();

        // check for empty matrices
        if (M == 0 || N == 0 || K == 0) return;

        // matrix mult
        for (std::size_t m = 0; m < M; ++m) 
        {
            for (std::size_t n = 0; n < N; ++n) 
            {
                float sum = 0.0f;
                if (inputs.size() > 2) 
                {
                    // broadcast bias if it exists 
                    if (inputs[2]->size() == N) 
                        sum = inputs[2]->data()[n];
                }

                for (std::size_t k = 0; k < K; ++k) 
                {
                    // handle transpositions
                    std::size_t a_idx = transA_ ? (k * M + m) : (m * K + k);
                    std::size_t b_idx = transpose_B ? (n * K + k) : (k * N + n);

                    sum += A_data[a_idx] * B_data[b_idx];
                }
                
                Y[m * N + n] = alpha_ * sum; // apply alpha
            }
        }
    }

private:
    float alpha_ = 1.0f;
    float beta_  = 1.0f;
    bool transA_ = false;
    bool transB_ = false;
};

#endif
