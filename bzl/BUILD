load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

exports_files([
    "conda.tpl.py",
    "conda_wrapper.py",
    "setup.tpl.py",
    "sphinx.py",
    "version.tpl.cc",
    "version.tpl.py",
])

config_setting(
    name = "with_callgrind",
    define_values = {"with_callgrind": "true"},
)

py_binary(
    name = "sphinx",
    srcs = [":sphinx.py"],
)

alias(
    name = "xgboost_alias",
    actual = select({
        "@com_intel_plaidml//toolchain:macos_x86_64": "@com_intel_plaidml_conda_unix//:xgboost_lib",
        "@com_intel_plaidml//toolchain:windows_x86_64": "@com_intel_plaidml_conda_windows//:xgboost_lib",
        "//conditions:default": "@com_intel_plaidml_conda_unix//:xgboost_lib",
    }),
)
