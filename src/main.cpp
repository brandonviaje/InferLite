#include <iostream>
#include <vector>
#include <algorithm>

#include "graph.h"
#include "onnx_parser.h"
#include "image_loader.h"
#include "inference_engine.h"

int main(int argc, char** argv)
{
    if (argc < 3) 
    {
        std::cerr << "Usage: ./infera <model.onnx> <image.png>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    try {
        // build the graph from the onnx model
        Graph graph;
        OnnxParser parser;

        std::cout << "Loading Model: " << model_path << "...\n";
        parser.parse(graph, model_path);

        // load input image into tensor
        std::cout << "Loading Image: " << image_path << "...\n";
        Tensor<float>* input_tensor = ImageLoader::load_image(image_path);

        InferenceEngine engine;
        std::cout << "Running Inference...\n";

        // engine expects inputs as a vector of tensor pointers
        std::vector<Tensor<float>*> inputs = { input_tensor };
        std::vector<Tensor<float>*> outputs = engine.run(graph, inputs);

        // grab output probabilities 
        if (outputs.empty())
            throw std::runtime_error("No output from engine.");

        Tensor<float>* result = outputs[0];
        const float* output_data = result->data();

        int predicted_digit = -1;
        float max_prob = -1e9f;

        std::cout << "\n=== Results ===\n";
        for (int i = 0; i < 10; ++i) 
        {
            std::cout << "Digit " << i << ": " << output_data[i] << "\n";

            // manual argmax
            if (output_data[i] > max_prob) 
            {
                max_prob = output_data[i];
                predicted_digit = i;
            }
        }

        std::cout << "PREDICTION: " << predicted_digit << "\n";

        delete input_tensor; 
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
