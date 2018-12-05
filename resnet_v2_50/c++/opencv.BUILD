cc_library(
    name = "opencv",
    srcs = glob(["lib/*.so*"]),
    hdrs = glob([ "include/opencv2/**/*.h", "include/opencv2/**/*.hpp", ]), 
    includes = ["include"],
    visibility = ["//visibility:public"], 
    linkstatic = 1,
)
