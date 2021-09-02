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