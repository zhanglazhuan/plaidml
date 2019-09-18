package(default_visibility = ["//visibility:public"])

exports_files(["env"])

filegroup(
    name = "bison",
    srcs = ["env/bin/bison"],
)

filegroup(
    name = "flex",
    srcs = ["env/bin/flex"],
)

filegroup(
    name = "python",
    srcs = ["env/bin/python"],
)

cc_library(
    name = "python_headers",
    hdrs = glob(["env/include/python3.7m/**/*.h"]),
    includes = ["env/include/python3.7m"],
)

cc_library(
    name = "pytorch",
    srcs = select({
        "@com_intel_plaidml//toolchain:macos_x86_64": [
            "env/lib/python3.7/site-packages/torch/lib/libc10.dylib",
            "env/lib/python3.7/site-packages/torch/lib/libtorch.dylib",
        ],
        "//conditions:default": [
            "env/lib/python3.7/site-packages/torch/lib/libc10.so",
            "env/lib/python3.7/site-packages/torch/lib/libgomp-8bba0e50.so.1",
            "env/lib/python3.7/site-packages/torch/lib/libtorch.so",
        ],
    }),
    includes = [
        "env/lib/python3.7/site-packages/torch/include",
    ],
    deps = [
        ":python_headers",
        "@pybind11",
    ],
)

cc_import(
    name = "xgboost_import",
    shared_library = select({
        "@com_intel_plaidml//toolchain:macos_x86_64": "env/lib/libxgboost.dylib",
        "//conditions:default": "env/lib/libxgboost.so",
    }),
    hdrs = glob([
        "env/include/xgboost/**",
    ]),
)

cc_library(
    name = "xgboost_lib",
    hdrs = glob([
       "env/include/xgboost/**",
    ]),
    includes = [
       "env/include",
       "env/include/xgboost",
    ],
    deps = [
        ":xgboost_import",
    ],
    visibility = ["//visibility:public"],
)
