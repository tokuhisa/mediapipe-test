#include "hand_tracking_cpu_lib.h"

float count = 0;
CPPLIBRARY_API float count_up(void)
{
	count += 0.1;
	return count;
}

CPPLIBRARY_API void count_init(void)
{
	count = 0;
}


CPPLIBRARY_API void test_uint8_array(uint8* data) {

}

CPPLIBRARY_API void test_float_array(float* data) {
	
}