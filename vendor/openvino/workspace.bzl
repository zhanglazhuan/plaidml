# Copyright 2020 Intel Corporation

load("//vendor/bazel:repo.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def openvino_workspace():
    http_archive(
        name = "ade",
        url = "https://github.com/opencv/ade/archive/cbe2db61a659c2cc304c3837406f95c39dfa938e.zip",
        strip_prefix = "ade-cbe2db61a659c2cc304c3837406f95c39dfa938e",
        sha256 = "6660e1b66bd3d8005026155571a057765ace9b0fdd9899aaa5823eca12847896",
        build_file = clean_dep("//vendor/openvino:ade.BUILD"),
    )

    http_archive(
        name = "openvino",
        url = "https://github.com/plaidml/openvino/archive/854851a93f2769cb8a59603cafa6c08f4cc7efb0.zip",
        strip_prefix = "openvino-854851a93f2769cb8a59603cafa6c08f4cc7efb0",
        sha256 = "1f62a191bd48efefcdcf2809c81cb0571bb6cd06ae42cb458c7baa3743e45815",
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
