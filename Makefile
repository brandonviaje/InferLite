# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Isrc

# Directories
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build

# Files
TEST_SRC = $(TEST_DIR)/tensor_test.cpp
TARGET = $(BUILD_DIR)/run_tests

# build and run
all: $(TARGET)
	./$(TARGET)

# create build directory and compile test
$(TARGET): $(TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $(TARGET)

# clean up build files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean