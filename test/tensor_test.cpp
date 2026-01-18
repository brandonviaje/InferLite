#include <iostream>
#include <cassert>
#include <vector>
#include "../src/tensor.h"

void test_constructor_and_size()
{
    std::vector<size_t> shape = {2, 3}; // 2x3 matrix
    Tensor<int> t(shape);

    assert(t.size() == 6);
    std::cout << "Constructor test passed!" << std::endl;
}

void test_copy_assignment()
{
    Tensor<float> t1({2, 2});
    Tensor<float> t2({3, 3});

    t2 = t1; // test copy assignment

    // check if metadata updated
    assert(t2.rows() == 2);
    std::cout << "Copy assignment test passed!" << std::endl;
}

void test_move_semantics()
{
    Tensor<int> t1({10, 10});
    Tensor<int> t2 = std::move(t1); // test move constructor

    // t1 should now be empty/null
    std::cout << "Move semantics test passed!" << std::endl;
}

void test_dimensions()
{
    Tensor<float> t({5, 3}); // 5 rows, 3 columns
    assert(t.rows() == 5);
    assert(t.cols() == 3);
    assert(t.size() == 15);
    std::cout << "Dimension tests passed!" << std::endl;
}

int main()
{
    try
    {
        test_constructor_and_size();
        test_copy_assignment();
        test_move_semantics();
        test_dimensions();
        std::cout << "\nALL TESTS PASSED!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}