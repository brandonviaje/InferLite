# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Isrc
LDFLAGS = -lprotobuf  

# Directories
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build

# Source Files
PROTO_SRC = $(SRC_DIR)/onnx-ml.pb.cc

$(SRC_DIR)/onnx-ml.pb.cc: proto/onnx-ml.proto
	@echo "Generating Protobuf files..."
	protoc --proto_path=proto --cpp_out=$(SRC_DIR) $<

# Targets
TENSOR_TEST_EXE = $(BUILD_DIR)/run_tensor_tests
MODEL_TEST_EXE  = $(BUILD_DIR)/run_model_tests

all: test

# Run both test suites
test: $(TENSOR_TEST_EXE) $(MODEL_TEST_EXE)
	@echo "--- Running Tensor Tests ---"
	@./$(TENSOR_TEST_EXE)
	@echo "\n--- Running Model Tests ---"
	@./$(MODEL_TEST_EXE)

# Build and Compile Tests 
$(TENSOR_TEST_EXE): $(TEST_DIR)/tensor_test.cpp
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) $< -o $@

$(MODEL_TEST_EXE): $(TEST_DIR)/model_test.cpp $(PROTO_SRC)
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Clean
clean:
	@rm -rf $(BUILD_DIR)

.PHONY: all test clean