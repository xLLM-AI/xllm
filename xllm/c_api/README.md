### How to compile xllm dynamic library

The C ABI is version 1. `XLLM_InitOptions` now exposes `cp_size` with a default
of `1`. This is an ABI-breaking structure change, so applications must be
recompiled against the installed version 1 headers and library.

MLU Context Parallel is available only for supported text-generation models.
Set `cp_size` to the global world size. In disaggregated PD deployments, set it
to `N` for the Prefill instance and `1` for the Decode instance.

Run the following command in root directory:

```
python setup.py build --generate-so true
```

If you want to debug, it needs to set DEBUG environment variable.

```
export DEBUG=1
```

### How to install dynamic library

Run installation script xllm/c_api/install.sh, headers and dynamic library will be installed in /usr/local/xllm directory.

```
cd xllm/c_api/tools

sh install.sh
```

You will see the following files in /usr/local/xllm directory:

```
[root@A03-R40-I189-101-4100046]# tree /usr/local/xllm
/usr/local/xllm
|-- include
|   |-- llm.h
|   |-- default.h
|   |-- rec.h
|   `-- types.h
`-- lib
    `-- libxllm.so

3 directories, 5 files
```

### How to compile c_api examples

GPU builds and NPU builds use different link commands. Replace
`<example>.cpp` and `<example>` with the example source file and output binary
name you want to build.

#### GPU

```
cd xllm/c_api/examples
g++ <example>.cpp -o <example> \
  -std=c++17 \
  -DUSE_CUDA \
  -I/usr/local/xllm/include \
  -L/usr/local/xllm/lib \
  -lxllm \
  -Wl,-rpath=/usr/local/xllm/lib
```

#### NPU

Before compiling or running examples, source the Ascend environment first:

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Then compile with the extra custom op library used by the NPU build:

```
cd xllm/c_api/examples
g++ <example>.cpp -o <example> \
  -std=c++17 \
  -DUSE_NPU \
  -I/usr/local/xllm/include \
  -L/usr/local/xllm/lib \
  -L/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/lib \
  -lxllm \
  -lcust_opapi \
  -Wl,-rpath=/usr/local/xllm/lib \
  -Wl,-rpath=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/lib
```


If `-lcust_opapi` is missing from the NPU link command, the linker may report
undefined references to symbols such as `aclnnBeamSearchGroup` and
`aclnnXAttention`.

### How to run c_api examples

Some examples, such as `simple_rec_completions`, support overriding the target
device from `argv[1]`.

#### NPU

```
./simple_rec_completions npu:14
```

#### GPU

```
./simple_rec_completions cuda:0
```
