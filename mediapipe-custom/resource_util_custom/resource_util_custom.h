#pragma once

#define CPPLIBRARY_API __declspec(dllexport) 


#include <csetjmp>
#include <csignal>
#include <string>
#include "mediapipe/util/resource_util_custom.h"


extern "C"
{
	
    typedef bool ResourceProvider(const char* path, std::string* output);

    CPPLIBRARY_API void set_custom_global_resource_provider(ResourceProvider* resource_provider);
}