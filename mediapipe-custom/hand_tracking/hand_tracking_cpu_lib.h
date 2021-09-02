#pragma once

#ifdef CPPLIBRARY_EXPORTS

#define CPPLIBRARY_API __declspec(dllexport) 

#else

#define CPPLIBRARY_API __declspec(dllimport) 

#endif

extern "C"
{
	CPPLIBRARY_API float count_up(void);
	CPPLIBRARY_API void count_init(void);
}