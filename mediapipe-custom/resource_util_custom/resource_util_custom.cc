#include "resource_util_custom.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/deps/ret_check.h"

CPPLIBRARY_API void set_custom_global_resource_provider(ResourceProvider* resource_provider) {
  mediapipe::SetCustomGlobalResourceProvider([resource_provider](const std::string& path, std::string* output) -> ::absl::Status {
    if (resource_provider(path.c_str(), output)) {
      return absl::OkStatus();
    }
    return absl::FailedPreconditionError(absl::StrCat("Failed to read ", path));
  });
}
