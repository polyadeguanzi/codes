target("gemm")
    set_kind("binary")
    add_files("gemm.cu")
target_end()

target("cat")
    set_kind("binary")
    add_files("cat.cu")
target_end()

target("cat")
    set_kind("binary")
    add_files("cat.cu")
target_end()

target("unfuse")
    set_kind("binary")
    add_files("unfuse.cu")

target("fuse_cg")
    set_kind("binary")
    add_files("fuse_cg.cu")
