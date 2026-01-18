# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Isrc

# Directories
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build

# Files
TEST_SRC = $(TEST_DIR)/tensor_test.cpp

# Targets
TEST_TARGET = $(BUILD_DIR)/run_tests

# build and run
all: test

test: $(TEST_TARGET)
	./$(TEST_TARGET)

# create build directory and compile test
$(TEST_TARGET): $(TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $(TEST_TARGET)

run:
	@echo "No main application yet.

# clean up build files
clean:
	@rm -rf $(BUILD_DIR)

.PHONY: all test run clean