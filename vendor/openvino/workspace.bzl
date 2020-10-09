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

   # http_archive(
   #     name = "openvino",
   #     url = "https://github.com/plaidml/openvino/archive/62963cfd8cbd0466c6bba56407250747193993e3.zip",
   #     strip_prefix = "openvino-62963cfd8cbd0466c6bba56407250747193993e3",
   #     sha256 = "507982d08519f53b17bd0c819d3cda1e837567e20a5d9aa0cc24304e0892de12",
   #     build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
   # )

    new_git_repository(
        name = "openvino",
        # sha256 = "40652941587e579d45a190731960008827221d11575f7f2e6162285b6625b940",
        remote = "file:///home/nchoudhu/github_repo/openvino/.git",
        commit = "87fbc738729e6e846e2fbeb282d56c8fdcb48132",
        #init_submodules = 1,
	build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
