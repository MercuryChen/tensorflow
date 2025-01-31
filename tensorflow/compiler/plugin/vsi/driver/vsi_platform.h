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

#ifndef TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VSI_PLATFORM_H_
#define TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VSI_PLATFORM_H_

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/vsi/driver/vsi_utils.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_platform_id.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/plugin.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/trace_listener.h"

#include "tim/vx/context.h"

namespace xla{
namespace vsiplugin{

namespace se = stream_executor;
namespace port = se::port;

class VsiPlatform : public se::Platform{
 public:
  VsiPlatform();
  ~VsiPlatform() override;

  Platform::Id id() const override;

  int VisibleDeviceCount() const override;

  const std::string& Name() const override;

  std::shared_ptr<tim::vx::Context> getContext() { return kVsiContext;}
  
  port::StatusOr<std::unique_ptr<se::DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  port::StatusOr<se::StreamExecutor*> ExecutorForDevice(int ordinal) override;

  port::StatusOr<se::StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const se::PluginConfig& config) override;

  port::StatusOr<se::StreamExecutor*> GetExecutor(
      const se::StreamExecutorConfig& config) override;

  port::StatusOr<std::unique_ptr<se::StreamExecutor>> GetUncachedExecutor(
      const se::StreamExecutorConfig& config) override;

  void RegisterTraceListener(std::unique_ptr<se::TraceListener> listener) override;

  void UnregisterTraceListener(se::TraceListener* listener) override;

 private:
  // This platform's name.
  std::string name_ = "vsi-npu";
  // This platform's id.
  Platform::Id id_ = kVsiPlatformId;

  // Cache of created StreamExecutors.
  se::ExecutorCache executor_cache_;

  std::shared_ptr<tim::vx::Context> kVsiContext;

  SE_DISALLOW_COPY_AND_ASSIGN(VsiPlatform);
};

} // namespace vsiplugin
} // namespace xla

#endif