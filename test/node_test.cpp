#include <iostream>
#include <cassert>
#include <vector>
#include "../src/node.h"      
#include "../src/onnx-ml.pb.h" 

void test_node_attributes() {
    std::cout << "Running Node Attribute Test...\n";

    // create mock ONNX node proto
    onnx::NodeProto node_proto;
    node_proto.set_name("TestConvNode");
    node_proto.set_op_type("Conv");

    // Add INT attr
    auto* attr_int = node_proto.add_attribute();
    attr_int->set_name("group");
    attr_int->set_type(onnx::AttributeProto::INT);
    attr_int->set_i(1);

    // Add FLOAT attr 
    auto* attr_float = node_proto.add_attribute();
    attr_float->set_name("alpha");
    attr_float->set_type(onnx::AttributeProto::FLOAT);
    attr_float->set_f(0.5f);

    // Add INTS attr
    auto* attr_ints = node_proto.add_attribute();
    attr_ints->set_name("pads");
    attr_ints->set_type(onnx::AttributeProto::INTS);
    attr_ints->add_ints(1);
    attr_ints->add_ints(1);
    attr_ints->add_ints(1);
    attr_ints->add_ints(1);

    // Add STRING attr
    auto* attr_string = node_proto.add_attribute();
    attr_string->set_name("auto_pad");
    attr_string->set_type(onnx::AttributeProto::STRING);
    attr_string->set_s("SAME_UPPER");

    Node node(node_proto);

    // assert

    // test INT retrievals
    auto group = node.get_attribute<int64_t>("group");
    assert(group.has_value());
    assert(group.value() == 1);
    std::cout << "  [PASS] INT Attribute\n";

    // test FLOAT retrieval
    auto alpha = node.get_attribute<float>("alpha");
    assert(alpha.has_value());
    assert(alpha.value() == 0.5f);
    std::cout << "  [PASS] FLOAT Attribute\n";

    // test INTS (Vector) retrieval
    auto pads = node.get_attribute<std::vector<int64_t>>("pads");
    assert(pads.has_value());
    assert(pads.value().size() == 4);
    assert(pads.value()[0] == 1);
    std::cout << "  [PASS] INTS Attribute\n";

    // test STRING retrieval
    auto auto_pad = node.get_attribute<std::string>("auto_pad");
    assert(auto_pad.has_value());
    assert(auto_pad.value() == "SAME_UPPER");
    std::cout << "  [PASS] STRING Attribute\n";

    // test MISSING Attribute
    auto missing = node.get_attribute<int64_t>("non_existent_attr");
    assert(!missing.has_value());
    std::cout << "  [PASS] Missing Attribute handled correctly\n";

    // test WRONG TYPE
    auto wrong_type = node.get_attribute<int64_t>("alpha");
    assert(!wrong_type.has_value()); 
    std::cout << "  [PASS] Wrong Type handled correctly\n";

}

int main() {
    try {
        test_node_attributes();
        std::cout << "\nNODE TESTS PASSED!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test Failed with Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
