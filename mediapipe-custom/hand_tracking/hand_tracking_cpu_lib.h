#pragma once

#define CPPLIBRARY_API __declspec(dllexport) 

#include <cstdlib>
#include "mediapipe/framework/deps/safe_int.h"


extern "C"
{
	CPPLIBRARY_API float count_up(void);
	CPPLIBRARY_API void count_init(void);
	CPPLIBRARY_API void test_uint8_array(uint8* data);
	CPPLIBRARY_API void test_float_array(float* data);
}