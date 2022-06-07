/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"

#include <memory.h>

#include "tensorflow/compiler/plugin/vsi/driver/vsi_utils.h"
#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/host/host_timer.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops.h"
#include "tim/vx/tensor.h"

namespace xla {
namespace vsiplugin {

const int invalid_index = 0x7fffff;

#if THRIFT_RPC
using namespace apache::thrift::transport;
using namespace apache::thrift::protocol;
using namespace shared;

#if 0
int rpc_demo() {
   /***********************generate client****************************/
    // transport layer
    std::shared_ptr<TTransport> socket(new TSocket(boss_socket_client(8080, 0)));
    std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    // protocol layer
    std::shared_ptr<TProtocol> mProtocol(new TBinaryProtocol(transport));
    std::string serviceName = "TrainingDemo";
    // use multilexedprotol
    std::shared_ptr<TMultiplexedProtocol> protocol(new TMultiplexedProtocol(mProtocol, serviceName));
    // create client
    shared_ptr<shared::RemoteClientClient> client(new shared::RemoteClientClient(protocol, protocol));
    transport->open();
    /***********************generate graph*****************************/
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    tim::vx::ShapeType a_shape({1, 2});
    tim::vx::ShapeType out_shape({1, 2});

    tim::vx::TensorSpec a_spec(tim::vx::DataType::FLOAT32,
                               a_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                                 out_shape, tim::vx::TensorAttribute::OUTPUT);
    auto a_tensor = graph->CreateTensor(a_spec);
    auto out_tensor = graph->CreateTensor(out_spec);
    auto op_relu = graph->CreateOperation<tim::vx::ops::Relu>();
    (*op_relu).BindInputs({a_tensor}).BindOutputs({out_tensor});

    uint32_t data_size = 2;
    std::vector<float> input_a_data{3,-1};
    std::vector<float> output_data(data_size);
    int64_t tensor_size = sizeof(float)*data_size;
    /***********************run code******************************/
    int32_t device_handles = client->Enumerate();
    int32_t device_handle = device_handles - 1;
    std::shared_ptr<tim::vx::platform::IDevice> remote_device= std::make_shared<tim::vx::platform::RemoteDevice>(client,device_handle);
    auto remote_executor= std::make_shared<tim::vx::platform::RemoteExecutor>(remote_device);
    auto remote_exectable = remote_executor->Compile(graph);

    auto input_tensor = remote_exectable->AllocateTensor(a_spec);
    input_tensor->CopyDataToTensor(input_a_data.data(),tensor_size);
    remote_exectable->SetInput(input_tensor);

    auto output_tensor = remote_exectable->AllocateTensor(out_spec);
    output_tensor->CopyDataToTensor(output_data.data(),tensor_size);
    remote_exectable->SetOutput(output_tensor);

    remote_exectable->Submit(remote_exectable);

    remote_executor->Trigger(true);
    std::cerr<<"##### end code"<<std::endl;

    std::vector<float> result(data_size);
    remote_exectable->GetOutput({output_tensor});
    output_tensor->CopyDataFromTensor(result.data());
    std::cerr<<"#####final result is "<<result[0]<<" "<<result[1]<<std::endl;
    return 1;
}
#endif
#endif

VsiExecutor::VsiExecutor(std::shared_ptr<tim::vx::Context> vsiCtx,
                         const int device_ordinal,
                         se::PluginConfig pluginConfig)
    : kVsiContext(vsiCtx), ordinal_(device_ordinal), plugConfig_(pluginConfig) {
  std::unique_lock<std::mutex> lock(mutex_);
  LOG(INFO) << __FUNCTION__ << " UUU 0";
  // kVsiGraphContainer[ordinal_] = kVsiContext->CreateGraph();
#if THRIFT_RPC
  std::string serviceName = "TrainingDemo";
  socket_ = std::make_shared<TSocket>(boss_socket_client(8080, 0));
  transport_ = std::make_shared<TBufferedTransport>(socket_);
  protocol_ = std::make_shared<TBinaryProtocol>(transport_);
  multiplexed_protocol_ =
      std::make_shared<TMultiplexedProtocol>(protocol_, serviceName);
  client_ = std::make_shared<shared::RemoteClientClient>(multiplexed_protocol_,
                                                         multiplexed_protocol_);
  transport_->open();

  int32_t device_handles = client_->Enumerate();
  int32_t device_handle = device_handles - 1;
  remote_device_ =
      std::make_shared<tim::vx::platform::RemoteDevice>(client_, device_handle);
  remote_executor_ =
      std::make_shared<tim::vx::platform::RemoteExecutor>(remote_device_);

  // rpc_demo();
#endif
}

VsiExecutor::~VsiExecutor() {
  LOG(INFO) << __FUNCTION__ << " UUU X";
#if THRIFT_RPC
  remote_executor_->Clear();
  transport_->close();
#endif
}

// TODO: temprarily use 1d tensor
se::DeviceMemoryBase VsiExecutor::Allocate(uint64 size, int64 memory_space) {
  // tim::vx::ShapeType input_shape({size});
  // tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 1.0f, 0);
  // tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
  //                                 tim::vx::TensorAttribute::VARIABLE,
  //                                 input_quant);
  //
  // kVsiTensorContainer.push_back(
  // kVsiGraphContainer[ordinal_]->CreateTensor(input_spec) );
  std::unique_lock<std::mutex> lock(mutex_);
  void* data = malloc(size);
  return se::DeviceMemoryBase(data, size);
  // return se::DeviceMemoryBase( kVsiTensorContainer.back().get(), size);
}

void* VsiExecutor::GetSubBuffer(se::DeviceMemoryBase* parent, uint64 offset,
                                uint64 size) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return nullptr;
}

void VsiExecutor::Deallocate(se::DeviceMemoryBase* mem) {
  std::unique_lock<std::mutex> lock(mutex_);
  free(mem->opaque());
}

void* VsiExecutor::HostMemoryAllocate(uint64 size) {
  void* ptr = malloc(size);
  return ptr;
}
void VsiExecutor::HostMemoryDeallocate(void* mem) { free(mem); }
bool VsiExecutor::HostMemoryRegister(void* mem, uint64 size) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
}
bool VsiExecutor::HostMemoryUnregister(void* mem) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
}
bool VsiExecutor::SynchronizeAllActivity() { return true; }

port::Status VsiExecutor::SynchronousMemZero(se::DeviceMemoryBase* location,
                                             uint64 size) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return port::Status::OK();
}
port::Status VsiExecutor::SynchronousMemSet(se::DeviceMemoryBase* location,
                                            int value, uint64 size) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return port::Status::OK();
}
port::Status VsiExecutor::SynchronousMemcpy(se::DeviceMemoryBase* gpu_dst,
                                            const void* host_src, uint64 size) {
  auto t = gpu_dst->opaque();
  if (host_src == nullptr) {
    LOG(FATAL) << "The ponit is nullprt, Something wrong !!";
  } else {
    memcpy(t, host_src, size);
  }
  // auto t = static_cast<tim::vx::Tensor *>(gpu_dst->opaque());
  // if(t != nullptr && size > 0)
  //     t->CopyDataToTensor(host_src, size);
  return port::Status::OK();
}
port::Status VsiExecutor::SynchronousMemcpy(void* host_dst,
                                            const se::DeviceMemoryBase& gpu_src,
                                            uint64 size) {
  auto t = gpu_src.opaque();
  if (t == nullptr) {
    LOG(FATAL) << "The ponit is nullprt, Something wrong !!";
  } else {
    memcpy(host_dst, t, size);
  }
  // auto t = static_cast<tim::vx::Tensor*>(const_cast<void
  // *>(gpu_src.opaque())); if(t != nullptr && size > 0)
  //     t->CopyDataFromTensor(host_dst);
  return port::Status::OK();
}
port::Status VsiExecutor::SynchronousMemcpyDeviceToDevice(
    se::DeviceMemoryBase* gpu_dst, const se::DeviceMemoryBase& gpu_src,
    uint64 size) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return port::Status::OK();
}
port::Status VsiExecutor::MemZero(se::Stream* stream,
                                  se::DeviceMemoryBase* location, uint64 size) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return port::Status::OK();
}
port::Status VsiExecutor::Memset32(se::Stream* stream,
                                   se::DeviceMemoryBase* location,
                                   uint32 pattern, uint64 size) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return port::Status::OK();
}

bool VsiExecutor::Memcpy(se::Stream* stream, void* host_dst,
                         const se::DeviceMemoryBase& gpu_src, uint64 size) {
  AsVsiStream(stream)->EnqueueTask([this, host_dst, gpu_src, size]() {
    auto ok = SynchronousMemcpy(host_dst, gpu_src, size);
  });
  AsVsiStream(stream)->BlockUntilDone();
  return true;
}
bool VsiExecutor::Memcpy(se::Stream* stream, se::DeviceMemoryBase* gpu_dst,
                         const void* host_src, uint64 size) {
  AsVsiStream(stream)->EnqueueTask([this, &gpu_dst, &host_src, size]() {
    auto ok = SynchronousMemcpy(gpu_dst, host_src, size);
  });
  AsVsiStream(stream)->BlockUntilDone();
  return true;
}

bool VsiExecutor::MemcpyDeviceToDevice(se::Stream* stream,
                                       se::DeviceMemoryBase* gpu_dst,
                                       const se::DeviceMemoryBase& gpu_src,
                                       uint64 size) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
}

se::host::HostStream* VsiExecutor::AsVsiStream(se::Stream* stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<se::host::HostStream*>(stream->implementation());
}

bool VsiExecutor::HostCallback(se::Stream* stream,
                               std::function<void()> callback) {
  // TENSORFLOW_TRACEPOINT();
  AsVsiStream(stream)->EnqueueTask(callback);
  return true;
}
bool VsiExecutor::HostCallback(se::Stream* stream,
                               std::function<port::Status()> callback) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
}
port::Status VsiExecutor::AllocateEvent(se::Event* event) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return port::Status::OK();
}
port::Status VsiExecutor::DeallocateEvent(se::Event* event) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return port::Status::OK();
}
port::Status VsiExecutor::RecordEvent(se::Stream* stream, se::Event* event) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return port::Status::OK();
}
port::Status VsiExecutor::WaitForEvent(se::Stream* stream, se::Event* event) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return port::Status::OK();
}
se::Event::Status VsiExecutor::PollForEventStatus(se::Event* event) {
  return se::Event::Status::kError;
}

bool VsiExecutor::AllocateStream(se::Stream* stream) { return true; }
void VsiExecutor::DeallocateStream(se::Stream* stream) { return; }
bool VsiExecutor::CreateStreamDependency(se::Stream* dependent,
                                         se::Stream* other) {
  AsVsiStream(dependent)->EnqueueTask(
      [other]() { auto ok = other->BlockHostUntilDone(); });
  AsVsiStream(dependent)->BlockUntilDone();
  return true;
}
bool VsiExecutor::AllocateTimer(Timer* timer) { return true; }
void VsiExecutor::DeallocateTimer(Timer* timer) { return; }
bool VsiExecutor::StartTimer(se::Stream* stream, Timer* timer) {
  dynamic_cast<se::host::HostTimer*>(timer->implementation())->Start(stream);
  return true;
}
bool VsiExecutor::StopTimer(se::Stream* stream, Timer* timer) {
  dynamic_cast<se::host::HostTimer*>(timer->implementation())->Stop(stream);
  return true;
}

port::Status VsiExecutor::BlockHostUntilDone(se::Stream* stream) {
  AsVsiStream(stream)->BlockUntilDone();
  return port::Status::OK();
}
int VsiExecutor::PlatformDeviceCount() {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
}
port::Status VsiExecutor::EnablePeerAccessTo(StreamExecutorInterface* other) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return port::Status::OK();
}
bool VsiExecutor::CanEnablePeerAccessTo(StreamExecutorInterface* other) {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
}

port::StatusOr<std::unique_ptr<se::DeviceDescription>>
VsiExecutor::CreateDeviceDescription() const {
  se::internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  builder.set_name("vsi-npu");
  builder.set_device_memory_size(static_cast<uint64>(8) * 1024 * 1024 * 1024);

  return builder.Build();
}

std::unique_ptr<se::internal::EventInterface>
VsiExecutor::CreateEventImplementation() {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return nullptr;
}
std::unique_ptr<se::internal::KernelInterface>
VsiExecutor::CreateKernelImplementation() {
  LOG(FATAL) << __FUNCTION__ << " Not Implemented";
  return nullptr;
}
std::unique_ptr<se::internal::StreamInterface>
VsiExecutor::GetStreamImplementation() {
  return std::unique_ptr<se::internal::StreamInterface>(
      new se::host::HostStream(0));
}
std::unique_ptr<se::internal::TimerInterface>
VsiExecutor::GetTimerImplementation() {
  return std::unique_ptr<se::internal::TimerInterface>(
      new se::host::HostTimer());
}

}  // namespace vsiplugin
}  // namespace xla