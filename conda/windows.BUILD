package(default_visibility = ["//visibility:public"])

exports_files(["env"])

filegroup(
    name = "bison",
    srcs = ["env/Library/usr/bin/bison.exe"],
)

filegroup(
    name = "flex",
    srcs = ["env/Library/usr/bin/flex.exe"],
)

filegroup(
    name = "python",
    srcs = ["env/python.exe"],
)

cc_import(
    name = "xgboost_import",
    shared_library = "env/lib/libxgboost.dll",
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
