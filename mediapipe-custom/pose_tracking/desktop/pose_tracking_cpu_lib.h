#pragma once

#define CPPLIBRARY_API __declspec(dllexport) 

extern "C"
{
	CPPLIBRARY_API float count_up(void);
	CPPLIBRARY_API void count_init(void);
}