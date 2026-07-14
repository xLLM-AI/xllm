### How to compile xllm dynamic library

`XLLM_InitLLMOptions::cp_size` defaults to `1`. MLU Context Parallel requires a
supported text-generation model and the global world size. For disaggregated
PD, configure `N` on Prefill and `1` on Decode.

Run the following command in root directory:
```
python setup.py build --generate-so true
```

If you want to debug, it needs to set DEBUG environment variable.
```
export DEBUG=1
```

### How to install dynamic library
Run installation script xllm/cc_api/install.sh, headers and dynamic library will be installed in /usr/local/xllm directory.
```
cd xllm/cc_api

sh install.sh
```

You will see the following files in /usr/local/xllm directory:
```
[root@A03-R40-I189-101-4100046 cc_api]# tree /usr/local/xllm
/usr/local/xllm
|-- include
|   |-- llm.h
|   |-- macros.h
|   `-- types.h
`-- lib
    |-- libcust_opapi.so
    `-- libxllm.so

3 directories, 5 files
```

### How to run cc_api examples
It provides two examples which use cc_api to create xllm instance and run inference. The  single_llm_instance.cpp creates one instance which is used in most LLM scenes. The multiple_llm_instances.cpp creates two instances which is used in multiple-models scene or one model with multiple versions. 

You can follow the commands to compile and run these examples:
```
cd examples && mkdir build
cd build && cmake .. && make && cd ..

sh start-llm-instance.sh
```
