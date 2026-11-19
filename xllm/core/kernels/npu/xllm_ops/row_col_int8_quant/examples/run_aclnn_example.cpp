#include "acl/acl.h"
#include "aclnn_row_col_int8_quant.h"
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


class Tensor {
 public:
  Tensor(std::vector<int64_t> shape,aclDataType dtype,Buffer& buffer){
    std::vector<int64_t> stride(shape.size(),1);
    for(int i=static_cast<int>(shape.size())-2;i>=0;--i)stride[i]=stride[i+1]*shape[i+1];
    value=aclCreateTensor(shape.data(),shape.size(),dtype,stride.data(),0,ACL_FORMAT_ND,shape.data(),shape.size(),buffer.ptr());
    if(!value)throw std::runtime_error("aclCreateTensor failed");
  }
  ~Tensor(){if(value)aclDestroyTensor(value);}
  Tensor(const Tensor&)=delete;
  Tensor& operator=(const Tensor&)=delete;
  aclTensor* value=nullptr;
};

void Example(aclrtStream stream) {
  Buffer x(8),pr(64*4),pc(256*4),r(8),c(8),qr(4),qc(4);
  x.Write(std::vector<uint16_t>{0,0x4000,0xc400,0});
  Tensor tx({2,2},ACL_FLOAT16,x),tqr({2,2},ACL_INT8,qr),tqc({2,2},ACL_INT8,qc),tr({2},ACL_FLOAT,r),tc({2},ACL_FLOAT,c);
  uint64_t bytes=0;aclOpExecutor* executor=nullptr;
  Check(aclnnRowColInt8QuantGetWorkspaceSize(tx.value,tqr.value,tqc.value,tr.value,tc.value,&bytes,&executor));
  Buffer workspace(bytes);
  Check(aclnnRowColInt8Quant(workspace.ptr(),bytes,executor,stream));
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
    std::cout<<"ACLNN NPU example passed"<<std::endl;
    return 0;
  } catch(const std::exception& exc) {
    std::cerr<<exc.what()<<std::endl;
    if(stream) aclrtDestroyStream(stream);
    if(deviceSet) aclrtResetDevice(device);
    if(initialized) aclFinalize();
    return 1;
  }
}
