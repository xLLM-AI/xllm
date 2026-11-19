#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "aclnn/acl_meta.h"
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <atomic>
namespace {
std::atomic<int64_t> last_workspace_bytes{0};
struct Api {
 using Get=aclnnStatus(*)(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,uint64_t*,aclOpExecutor**);
 using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
 void* library;Get get;Execute execute;
 explicit Api(const std::string& path){
  library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
  TORCH_CHECK(library,"Cannot load generated ACLNN library: ",dlerror());
  get=reinterpret_cast<Get>(dlsym(library,"aclnnRowColInt8QuantGetWorkspaceSize"));
  execute=reinterpret_cast<Execute>(dlsym(library,"aclnnRowColInt8Quant"));
  TORCH_CHECK(get && execute,"Generated ACLNN functions missing");
 }
};

struct HostMemoryScope {
 using Init=int(*)(void*,bool);
 using Finish=void(*)(void*,bool);
 Init init=nullptr;Finish release=nullptr,uninit=nullptr;
 HostMemoryScope(){
  static void* lib=dlopen("libopapi.so",RTLD_NOW|RTLD_LOCAL);
  static auto begin=lib?reinterpret_cast<Init>(dlsym(lib,"InitHugeMemThreadLocal")):nullptr;
  static auto end=lib?reinterpret_cast<Finish>(dlsym(lib,"ReleaseHugeMem")):nullptr;
  static auto close=lib?reinterpret_cast<Finish>(dlsym(lib,"UnInitHugeMemThreadLocal")):nullptr;
  if(begin && end && close){init=begin;release=end;uninit=close;init(nullptr,false);}
 }
 ~HostMemoryScope(){if(release)release(nullptr,false);if(uninit)uninit(nullptr,false);}
};

struct TensorInfo {
 std::vector<int64_t> sizes,strides;
 aclDataType dtype;
 void* data;
 explicit TensorInfo(const at::Tensor& t):sizes(t.sizes().begin(),t.sizes().end()),strides(t.strides().begin(),t.strides().end()),data(t.data_ptr()){
  dtype=t.scalar_type()==at::kHalf?ACL_FLOAT16:t.scalar_type()==at::kBFloat16?ACL_BF16:t.scalar_type()==at::kChar?ACL_INT8:ACL_FLOAT;
 }
};
struct Metadata {
 std::vector<aclTensor*> tensors;
 ~Metadata(){for(auto* t:tensors)aclDestroyTensor(t);}
 void add(const TensorInfo& t){
  auto* result=aclCreateTensor(t.sizes.data(),t.sizes.size(),t.dtype,t.strides.data(),0,ACL_FORMAT_ND,t.sizes.data(),t.sizes.size(),t.data);
  TORCH_CHECK(result,"aclCreateTensor failed");tensors.push_back(result);
 }
};
void record(const std::vector<at::Tensor>& ts,c10_npu::NPUStream stream){
 for(const auto& t:ts)if(t.numel())c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
}
std::vector<at::Tensor> run_aclnn(const std::vector<at::Tensor>& in,double eps,const std::string& path){
  TORCH_CHECK(in.size() == 1, "wrong input count");
  const auto& x = in[0];
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "NPU inputs required");
  for (const auto& t : in)
    TORCH_CHECK(t.device() == x.device() && t.scalar_type() == x.scalar_type() && t.is_contiguous(),
                "inputs must be contiguous and share device and dtype");
  TORCH_CHECK(x.scalar_type() == at::kHalf || false,
              "unsupported dtype");
  c10_npu::NPUGuard guard(x.device());
  auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  auto raw_stream = stream.stream(false);
  std::vector<at::Tensor> out, workspace;
  TORCH_CHECK(x.dim()==2,"x must be a matrix");
  auto m=x.size(0),n=x.size(1);
  TORCH_CHECK(m>=1 && m<=4096 && n>=2 && n<=4096 && n%2==0,"shape outside contract");
  auto rt=(m+63)/64,ct=(n+255)/256;
  out={at::empty(x.sizes(),x.options().dtype(at::kChar)),at::empty(x.sizes(),x.options().dtype(at::kChar)),
       at::empty({m},x.options().dtype(at::kFloat)),at::empty({n},x.options().dtype(at::kFloat))};


 static const Api api(path);
 std::vector<TensorInfo> infos;
 for(const auto& t:in)infos.emplace_back(t);
 for(const auto& t:out)infos.emplace_back(t);
 // Outputs and scratch are allocated on the current launch stream.
 // Its FIFO ordering already protects reuse. recordStream is required for
 // inputs from other streams; recording fresh outputs forces needless events.
 record(in,stream);
 // Allocating NPU memory inside a queue worker can wait for that same queue.
 // Preallocate the documented upper bound on the caller thread instead.
 // This equals the measured CANN 9.1.0 / Ascend910_9382 tiling requirement.
 const uint64_t capacity=16777216ULL+4ULL*((x.size(0)+63)/64)*((x.size(1)+255)/256)*320;
 at::Tensor scratch=at::empty({static_cast<int64_t>(capacity)},x.options().dtype(at::kByte));
 void* scratch_pointer=scratch.data_ptr();
 const auto device=x.device();
 auto get=api.get;auto execute=api.execute;
 auto submit=[=]() -> int {
  c10_npu::NPUGuard guard(device);
  HostMemoryScope memory_scope;
  Metadata descriptions;
  for(const auto& info:infos)descriptions.add(info);
  const auto& meta=descriptions.tensors;
  uint64_t bytes=0;aclOpExecutor* executor=nullptr;
  auto status=get(meta[0],meta[1],meta[2],meta[3],meta[4],&bytes,&executor);
  TORCH_CHECK(status==0,"aclnnRowColInt8QuantGetWorkspaceSize failed: ",status);
  last_workspace_bytes.store(static_cast<int64_t>(bytes));
  TORCH_CHECK(bytes<=capacity,"ACLNN workspace exceeds the validated hardware/CANN budget");
  status=execute(bytes?scratch_pointer:nullptr,bytes,executor,raw_stream);
  TORCH_CHECK(status==0,"aclnnRowColInt8Quant failed: ",status);
  return 0;
 };
 at_npu::native::OpCommand::RunOpApiV2("AclnnRowColInt8Quant",submit);
 return out;
}
}
TORCH_LIBRARY_FRAGMENT(cann_community,m){m.def("row_col_int8_quant_aclnn(Tensor[] inputs, float epsilon, str library) -> Tensor[]");}
TORCH_LIBRARY_IMPL(cann_community,PrivateUse1,m){m.impl("row_col_int8_quant_aclnn",TORCH_FN(run_aclnn));}

TORCH_LIBRARY_FRAGMENT(cann_community,m){m.def("row_col_int8_quant_workspace_bytes() -> int",[]() -> int64_t {return last_workspace_bytes.load();});}
