# Introduction

I was coding simple examples to learn CUDA concepts. It's always good to have
more example code available, so I figured I'd publish these.

# Running Examples

For C-examples, compile the .cu files with `nvcc`

`$ nvcc 0_hello_world.cu -o 0_hello_world.cu`

For Python examples, just run the file with python

`$ python 0_hello_world.py`

# Installing pyCUDA on Ubuntu 14.04


`$ sudo apt-get install python-numpy -y`

`$ sudo apt-get install build-essential python-dev python-setuptools libboost-python-dev libboost-thread-dev -y`

Download pyCUDA and unpack it

`$ tar xzvf pycuda-VERSION.tar.gz`

```
$ cd pycuda-VERSION
$ ./configure.py --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib --boost-python-libname=boost_python-mt --boost-thread-libname=boost_thread-mt
$ make
$ sudo make install
```
