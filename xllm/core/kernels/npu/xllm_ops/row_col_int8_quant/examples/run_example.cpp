#include "acl/acl.h"
#include "api.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void Check(aclError status) {
  if (status!=ACL_SUCCESS) throw std::runtime_error("ACL error "+std::to_string(status));
}
class Buffer {
 public:
  explicit Buffer(size_t bytes): bytes_(bytes) {
    if (bytes) Check(aclrtMalloc(&ptr_,bytes,ACL_MEM_MALLOC_HUGE_FIRST));
  }
  ~Buffer() { if(ptr_) aclrtFree(ptr_); }
  Buffer(const Buffer&)=delete;
  Buffer& operator=(const Buffer&)=delete;
  uint8_t* ptr(){return static_cast<uint8_t*>(ptr_);}
  template<class T> void Write(const std::vector<T>& data) {
    if(data.size()*sizeof(T)!=bytes_) throw std::runtime_error("input size mismatch");
    if(bytes_) Check(aclrtMemcpy(ptr_,bytes_,data.data(),bytes_,ACL_MEMCPY_HOST_TO_DEVICE));
  }
  template<class T> std::vector<T> Read() {
    std::vector<T> data(bytes_/sizeof(T));
    if(bytes_) Check(aclrtMemcpy(data.data(),bytes_,ptr_,bytes_,ACL_MEMCPY_DEVICE_TO_HOST));
    return data;
  }
 private: void* ptr_=nullptr; size_t bytes_;
};

void Example(aclrtStream stream) {
  Buffer x(8),pr(64*4),pc(256*4),r(8),c(8),qr(4),qc(4);
  x.Write(std::vector<uint16_t>{0,0x4000,0xc400,0});
  row_col_stats_launch(1,stream,x.ptr(),pr.ptr(),pc.ptr(),2,2);
  row_col_merge_launch(2,stream,pr.ptr(),pc.ptr(),r.ptr(),c.ptr(),2,2);
  row_col_quant_launch(1,stream,x.ptr(),r.ptr(),c.ptr(),qr.ptr(),qc.ptr(),2,2);
  Check(aclrtSynchronizeStream(stream));
  const std::vector<int8_t> expected{0,127,-127,0};
  if(qr.Read<int8_t>()!=expected||qc.Read<int8_t>()!=expected||
     r.Read<float>()!=std::vector<float>{2,4}||c.Read<float>()!=std::vector<float>{4,2})
    throw std::runtime_error("analytic quantization mismatch");
}

int main(int argc,char** argv) {
  int device=argc>1?std::stoi(argv[1]):0;
  aclrtStream stream=nullptr;
  bool initialized=false,deviceSet=false;
  try {
    Check(aclInit(nullptr));initialized=true;
    Check(aclrtSetDevice(device));deviceSet=true;
    Check(aclrtCreateStream(&stream));
    Example(stream);
    Check(aclrtDestroyStream(stream));stream=nullptr;
    Check(aclrtResetDevice(device));deviceSet=false;
    Check(aclFinalize());initialized=false;
    std::cout<<"NPU example passed"<<std::endl;
    return 0;
  } catch(const std::exception& exc) {
    std::cerr<<exc.what()<<std::endl;
    if(stream) aclrtDestroyStream(stream);
    if(deviceSet) aclrtResetDevice(device);
    if(initialized) aclFinalize();
    return 1;
  }
}
