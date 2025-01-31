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

#include "tensorflow/compiler/plugin/vsi/driver/vsi_executable.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace vsiplugin {

VsiExecutable::VsiExecutable(std::shared_ptr<HloModule> hlo_module,
                             VsiExecutor* executor)
    : Executable(hlo_module,
                 /*hlo_profile_printer_data=*/nullptr,
                 /*hlo_profile_index_map=*/nullptr),
      visitor_(std::move(std::make_unique<BaseVisitor>(executor))),
      executor_(executor) {
  visitor_->ResetVisitStates();
}

VsiExecutable::~VsiExecutable() {}

tensorflow::mutex vsi_executable_mtx;
StatusOr<ExecutionOutput> VsiExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  tensorflow::mutex_lock l(vsi_executable_mtx);
  LOG(INFO) << "ExecuteAsyncOnStream " << module().name()
            << " :: " << (void*)this
            << " :: " << tensorflow::Env::Default()->GetCurrentThreadId();

  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();
  const se::Platform* platform = executor->platform();

  // Convert the ShapeTree to a ShapedBuffer. We do this so we can call
  // TransferManager methods below.
  std::vector<ShapedBuffer> argument_buffers;
  argument_buffers.reserve(arguments.size());
  int device_ordinal = executor->device_ordinal();

  LOG(INFO) << "ExecuteAsyncOnStream VVV: " << device_ordinal;

  for (auto& argument : arguments) {
    const ShapeTree<MaybeOwningDeviceMemory>& buffers = argument.Buffers();
    argument_buffers.push_back(ShapedBuffer(buffers.shape(),
                                            /*device_ordinal=*/device_ordinal));
    auto in_it = buffers.begin();
    auto out_it = argument_buffers.back().buffers().begin();
    for (; in_it != buffers.end(); ++in_it, ++out_it) {
      out_it->second = in_it->second.AsDeviceMemoryBase();
    }
  }

  VLOG(1) << "Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : argument_buffers) {
      VLOG(2) << "-- argument " << a;
    }
  }

  const HloComputation* computation = module().entry_computation();
  if (computation->num_parameters() != arguments.size()) {
    return tensorflow::errors::Internal(
        "Mismatch between argument count and graph parameter count.");
  }

  TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager,
                      TransferManager::GetForPlatform(platform));
  // Transform the ShapedBuffer arguments into literals which the evaluator
  // consumes.
  std::vector<Literal> arg_literals;
  for (int64 p = 0; p < computation->num_parameters(); ++p) {
    TF_ASSIGN_OR_RETURN(Literal arg_literal,
                        transfer_manager->TransferLiteralFromDevice(
                            run_options->stream(), argument_buffers[p]));
    arg_literals.push_back(std::move(arg_literal));
  }
  LOG(INFO) << "computation->num_parameters: " << computation->num_parameters();

  auto tensor = visitor_->evaluate(*computation, arg_literals);

  // Transform the result literal back into a ShapedBuffer.
  auto root_instr = computation->root_instruction();
  const Shape& result_shape = root_instr->shape();
  LOG(INFO) << "ExecuteAsyncOnStream NNN 0: ";
  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer result_buffers,
      transfer_manager->AllocateScopedShapedBuffer(
          result_shape, run_options->allocator(), executor->device_ordinal()));
#if THRIFT_RPC
  LOG(INFO) << "GetOutput QQQ remote_outputs_.size: " << visitor_->remote_outputs_.size();
  visitor_->remote_exectable_->GetOutput(visitor_->remote_outputs_);
#endif

  LOG(INFO) << "GetOutput QQQ result_buffers: " << result_buffers.ToString();

  if (!result_shape.IsTuple()) {
    LOG(INFO) << "GetOutput QQQ 1: ";
#if THRIFT_RPC
    CHECK_EQ(visitor_->remote_outputs_.size(), 1);
#endif
    for (auto& pair : result_buffers.buffers()) {
      const ShapeIndex& index = pair.first;
      se::DeviceMemoryBase& memory_base = pair.second;
      const Shape& subshape =
          ShapeUtil::GetSubshape(result_buffers.on_device_shape(), index);
      LOG(INFO) << "no-tuple  result buffer info " << subshape.ToString();
#if THRIFT_RPC
      visitor_->remote_outputs_[0]->CopyDataFromTensor(memory_base.opaque());
#else
      tensor[0]->CopyDataFromTensor(memory_base.opaque());
#endif
      // float* val = (float*)(memory_base.opaque());
      // LOG(INFO) << "memory_base.opaque: " << *val;
    }
  } else {
    LOG(INFO) << "GetOutput QQQ 2: ";
    int32_t count = 0;
    auto top_shape_memory = result_buffers.buffers();

    se::DeviceMemoryBase top_memory_base;
    for (auto& pair : result_buffers.buffers()) {
      if (count == 0) {
        top_memory_base = pair.second;
        LOG(INFO) << "top_memory_base location is " << top_memory_base.opaque();
        // count++;
        break;
      }
    }

    count = 0;
    for (auto& pair : result_buffers.buffers()) {
      if (count == 0) {
        count++;
        continue;
      }
      const ShapeIndex& index = pair.first;
      se::DeviceMemoryBase& memory_base = pair.second;
      const Shape& subshape =
          ShapeUtil::GetSubshape(result_buffers.on_device_shape(), index);
      LOG(INFO) << "tuple result buffer info " << subshape.ToString();
#if THRIFT_RPC
      visitor_->remote_outputs_[count - 1]->CopyDataFromTensor(
          memory_base.opaque());
#else
      tensor[count - 1]->CopyDataFromTensor(memory_base.opaque());
#endif
      *(size_t*)(top_memory_base.opaque() + sizeof(void*) * (count - 1)) =
          (size_t)memory_base.opaque();
      LOG(INFO) << "sub tensor " << count << " mem is: " << memory_base.opaque();
      count++;
    }
  }

  ExecutionOutput result(std::move(result_buffers));
  LOG(INFO) << "Leave " << module().name() << " :: " << (void*)this
            << " :: " << tensorflow::Env::Default()->GetCurrentThreadId();
  return result;
}

StatusOr<std::vector<ScopedShapedBuffer>> VsiExecutable::ExecuteOnStreams(
    absl::Span<const ServiceExecutableRunOptions> run_options,
    absl::Span<const absl::Span<const ShapedBuffer* const>> arguments) {
  LOG(FATAL) << "not implement";
}

Status VsiExecutable::PopulateExecutionProfile(
    ExecutionProfile* execution_profile,
    HloExecutionProfile* hlo_execution_profile, se::Stream* stream) {
  LOG(FATAL) << "not implement";
}

}  // namespace vsiplugin
}  // namespace xla