//
// Created by dkafri on 10/13/20.
//

#include <iostream>

#ifndef PROTOTYPE_INCLUDE_TEST_UTILS_H_
#define PROTOTYPE_INCLUDE_TEST_UTILS_H_

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
