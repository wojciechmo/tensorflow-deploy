### Deploy TensorFlow models trained with Python using Java, C and C++. 

Three models:
- simple_graph: Python -> Java/C/C++
- resnet_v2_50: -> Java/C++
- big_gan_512: Python -> C

#### Train with TensorFlow Python API and save frozen model:
1. Install tensorflow: https://www.tensorflow.org/install<br />
2. Install tensorflow_hub: https://www.tensorflow.org/hub/<br />
3. Install opencv: https://pypi.org/project/opencv-python/<br />
```$ python train.py```<br />

#### Deploy using TensorFlow C++ API:
C++ API: https://www.tensorflow.org/guide/extend/cc<br/>
1. Build OpenCV from source: https://opencv.org/releases.html (-D CMAKE_INSTALL_PREFIX=/tmp/opencv-3.4/install) and add install directory to path: PATH="$PATH:/tmp/opencv-3.4/install"<br />
2. Install Bazel: https://docs.bazel.build/versions/master/install-ubuntu.html<br />
3. Clone TensorFlow GitHub repository: https://github.com/tensorflow/tensorflow<br />
4. Place BUILD and main.cpp files in tensorflow/cc/project directory<br />
5. Add the following to main repository WORKSPACE file:
```
new_local_repository(
    name = "opencv",
    path = "/tmp/opencv-3.4/install",
    build_file = "opencv.BUILD")
```
6. Place opencv.BUILD in the same directory as main repository WORKSPACE file with the following:
```
cc_library(
    name = "opencv",
    srcs = glob(["lib/*.so*"]),
    hdrs = glob([ "include/opencv2/**/*.h", "include/opencv2/**/*.hpp", ]), 
    includes = ["include"],
    visibility = ["//visibility:public"], 
    linkstatic = 1)
```
Then project can depend on @opencv//:opencv to link in the .so's under lib/ and reference the headers under include/.

7. Build from repository workspace:<br/>
```bazel build --jobs 6 --ram_utilization_factor 50 //tensorflow/cc/project:main```
8. Run from repository workspace:<br/>
```./bazel-bin/tensorflow/cc/project/main```

Note that when deploying model with tfhub module on remote computer, data from /tmp/tfhub_modules (or directory set with TFHUB_CACHE_DIR) must be copied under exactly the same absolute path in remote computer. It can't be done with manipulating TFHUB_CACHE_DIR env because tfhub directory absolute path is hardcoded inside model when it's saved.
