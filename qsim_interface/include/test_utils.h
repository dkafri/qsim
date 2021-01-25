//
// Created by dkafri on 10/13/20.
//

#include <iostream>

#ifndef PROTOTYPE_INCLUDE_TEST_UTILS_H_
#define PROTOTYPE_INCLUDE_TEST_UTILS_H_

#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            throw std::runtime_error("Assertion failed (see top error)."); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif


template<typename T, typename V>
bool almost_equals(const T& actual, const V& expected, float delta = 1e-6) {
  if (abs(actual - (T) expected) >= delta)
    return false;

  // in case of conversion errors
  return abs((V) actual - expected) < delta;
}

template<typename T, typename V>
bool equals(const T& actual, const V& expected) {
  if (actual != (T) expected)
    return false;

  // in case of conversion errors
  return (V) actual == expected;
}

#endif //PROTOTYPE_INCLUDE_TEST_UTILS_H_
