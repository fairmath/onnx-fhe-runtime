# Fairmath ONNX FHE Runtime

## Description


## How To Build
* Install CMake 3.22(or above), gcc or clang 
* Install OpenMP(this is not necessary but highly recommended)

### Option 1
Using OpenFHE installed on your system.

Clone repository with submodules.
Configure cmake and run build

```shell
$ mkdir .build
$ cd .build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_OPENFHE=On -DCMAKE_INSTALL_PREFIX=./install
$ make build && make install
```
Installation folder will contain onnx FHE runtime library and OpenFHE libs.

### Option 2
Using OpenFHE submodule.

Clone repository with submodules.
Configure cmake and run build

```shell
$ mkdir .build
$ cd .build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_OPENFHE=Off -DCMAKE_INSTALL_PREFIX=./install
$ make build && make install
```
Installation folder will contain onnx FHE runtime library only.


## How To Run Inference

Find the example of inference in example folder. There is two models, original and FHE based model. `inference.py` runs both models and print results.

To run script you need to have `openfhe-python` bindings, openfhe installed on your system. Also `onnx` and `onnxruntime` should be installed. 
To install [OpenFHE](https://github.com/openfheorg/openfhe-development) and [openfhe-python](https://github.com/openfheorg/openfhe-python) bindings please refer to origin github repositories.
To install `onnx` and `onnxruntime` you may use the following command

```shell
$ pip install onnx onnxruntime
```
Once everything is installed you may execute the inference:

```shell
$ python inference.py
```