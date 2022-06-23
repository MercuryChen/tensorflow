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


#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_platform.h"
#include "tensorflow/compiler/tf2xla/kernels/index_ops.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"

#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/kernels/no_op.h"

namespace tensorflow {
const char* const DEVICE_XLA_VSI_NPU = "NPU";
const char* const DEVICE_VSI_NPU_XLA_JIT = "XLA_NPU_JIT";
const char* const PLATFORM_NAME = "vsi-npu";

std::vector<DataType> GetVsiNpuSupportedTypes() {
  return {
    DT_UINT8, DT_QUINT8, DT_UINT16, DT_INT8, DT_QINT8, DT_INT16, DT_INT32,
    DT_QINT32, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
    DT_COMPLEX128, DT_BOOL, DT_BFLOAT16};
};

static bool OpFilter(KernelDef* kdef) {
  LOG(INFO) << "OpFilter : " << kdef->op();

  if (kdef->op() == "_Arg") return true;
  if (kdef->op() == "VarIsInitializedOp") return true;
  if (kdef->op() == "NoOp") return true;
  if (kdef->op() == "_Retval") return true;
  if (kdef->op() == "ResourceApplyGradientDescent") return true;
  if (kdef->op() == "ReadVariableOp") return true;
  if (kdef->op() == "AssignAddVariableOp") return true;

  if (kdef->op() == "Add") return true;
  if (kdef->op() == "AddV2") return true;
  if (kdef->op() == "BiasAdd") return true;
  if (kdef->op() == "BiasAddGrad") return true;
  if (kdef->op() == "Sub") return true;
  if (kdef->op() == "Sqrt") return true;
  if (kdef->op() == "Square") return true;
  if (kdef->op() == "Mul") return true;
  if (kdef->op() == "MatMul") return true;
  if (kdef->op() == "Conv2D") return true;
  if (kdef->op() == "Conv2DBackpropInput") return true;
  if (kdef->op() == "Conv2DBackpropFilter") return true;
  if (kdef->op() == "Relu") return true;
  if (kdef->op() == "ReluGrad") return true;
  if (kdef->op() == "BroadcastTo") return true;
  if (kdef->op() == "Sum") return true;
  if (kdef->op() == "Equal") return true;
  if (kdef->op() == "SoftmaxCrossEntropyWithLogits") return true;
  // if (kdef->op() == "Pack") return true;
  // if (kdef->op() == "ConcatV2") return true;
  // if (kdef->op() == "Reshape") return true;
  if (kdef->op() == "MaxPool") return true;
  // if (kdef->op() == "MaxPoolGrad") return true;

  // if (kdef->op() == "XlaDot") return true;
  // if (kdef->op() == "RealDiv") return true;

  // if (kdef->op() == "Angle") return false;
  // if (kdef->op() == "Complex") return false;
  // if (kdef->op() == "ComplexAbs") return false;
  // if (kdef->op() == "Conj") return false;
  // if (kdef->op() == "FFT") return false;
  // if (kdef->op() == "FFT2D") return false;
  // if (kdef->op() == "FFT3D") return false;
  // if (kdef->op() == "IFFT") return false;
  // if (kdef->op() == "IFFT2D") return false;
  // if (kdef->op() == "IFFT3D") return false;
  // if (kdef->op() == "Imag") return false;

  // if (kdef->op() == "MaxPool") return false;
  // if (kdef->op() == "MaxPoolV2") return false;
  // if (kdef->op() == "MaxPool3D") return false;
  // if (kdef->op() == "MaxPoolGrad") return false;
  // if (kdef->op() == "MaxPoolGradV2") return false;
  // if (kdef->op() == "MaxPool3DGrad") return false;
  // if (kdef->op() == "MaxPoolGradGrad") return false;
  // if (kdef->op() == "MaxPoolGradGradV2") return false;
  // if (kdef->op() == "MaxPool3DGradGrad") return false;

  // if (kdef->op() == "AvgPool") return false;
  // if (kdef->op() == "AvgPool3D") return false;
  // if (kdef->op() == "AvgPoolGrad") return false;
  // if (kdef->op() == "AvgPool3DGrad") return false;

  // if (kdef->op() == "Diag") return false;
  // if (kdef->op() == "DiagPart") return false;
  // //if (kdef->op() == "ReverseSequence") return false;
  // if (kdef->op() == "Range") return false;
  // if (kdef->op() == "LinSpace") return false;
  // if (kdef->op() == "DepthToSpace") return false;
  // if (kdef->op() == "SpaceToBatchND") return false;
  // if (kdef->op() == "SpaceToBatch") return false;
  // if (kdef->op() == "LRN") return false;
  // if (kdef->op() == "LRNGrad") return false;
  // if (kdef->op() == "XlaReduceWindow") return false;
  // if (kdef->op() == "Sum") return false;
  // if (kdef->op() == "Prod") return false;
  // if (kdef->op() == "Min") return false;
  // if (kdef->op() == "Max") return false;
  // if (kdef->op() == "Mean") return false;
  // if (kdef->op() == "All") return false;
  // if (kdef->op() == "Any") return false;
  // if (kdef->op() == "XlaReduce") return false;
  // if (kdef->op() == "Reduce") return false;
  // // if (kdef->op() == "Conv2DBackpropFilter") return false;

  // if (kdef->op() == "NonMaxSuppressionV4") return false;
  // if (kdef->op() == "Qr") return false;
  // if (kdef->op() == "Real") return false;
  // if (kdef->op() == "Cast") return false;
  // if (kdef->op() == "RandomUniform") return false;

  if (kdef->op() == "Assert") {
    AddDtypeToKernelDefConstraint("T", DT_STRING, kdef);
    return true;
  }
  if (kdef->op() == "Const") {
    AddDtypeToKernelDefConstraint("dtype", DT_STRING, kdef);
    return true;
  }
  if (kdef->op() == "Function" || kdef->op() == "Pipeline" ||
      kdef->op() == "PipelineStage" || kdef->op() == "PipelineStageBackward" ||
      kdef->op() == "ResourceUpdate" || kdef->op() == "MultiConv") {
    AddDtypeToKernelDefConstraint("Tin", DT_RESOURCE, kdef);
    AddDtypeToKernelDefConstraint("Tout", DT_RESOURCE, kdef);
    AddDtypeToKernelDefConstraint("Tin", DT_VARIANT, kdef);
    AddDtypeToKernelDefConstraint("Tout", DT_VARIANT, kdef);
    return true;
  }

  return false;
}

class VsiNpuDevice : public XlaDevice {
 public:
  VsiNpuDevice(const SessionOptions& options, const XlaDevice::Options& devopts)
      : XlaDevice(options, devopts) {
    UseGpuDeviceInfo();
  }

  virtual ~VsiNpuDevice() {}
};

class XlaVsiNpuDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;

  virtual Status ListPhysicalDevices(std::vector<string>* devices) override {
    //devices->push_back(absl::StrCat("/physical_device:", DEVICE_XLA_VSI_NPU, ":0"));
    devices->push_back(absl::StrCat("/physical_device:", DEVICE_XLA_VSI_NPU, ":0"));
    devices->push_back(absl::StrCat("/physical_device:", DEVICE_XLA_VSI_NPU, ":1"));
    return Status::OK();
  }
};

Status XlaVsiNpuDeviceFactory::CreateDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_VSI_NPU, DEVICE_VSI_NPU_XLA_JIT);
  (void)registrations;

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_VSI_NPU_XLA_JIT;
  registration.autoclustering_policy =
      XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.cluster_resource_variable_ops_unsafely = true;
  registration.cluster_stack_ops = true;
  registration.cluster_tensor_array_ops = true;
  registration.cluster_stateful_rng_ops = true;
  registration.cluster_control_trigger = true;
  registration.elide_assert_and_checknumerics = true;
  registration.cluster_variant_ops = true;
  registration.cluster_slow_ops = true;
  registration.cluster_inaccurate_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_VSI_NPU, registration);

  auto platform = se::MultiPlatformManager::PlatformWithName(PLATFORM_NAME);
  if (!platform.ok()) {
    return platform.status();
  }
  std::set<int> tmp;
  tmp.insert(0);
  tmp.insert(1);
  absl::optional<std::set<int>> vsi_npu_ids({tmp});
  for(int i = 0;i<2;i++){
    vsi_npu_ids->insert(i);
  }

  auto* p = static_cast<xla::vsiplugin::VsiPlatform*>(platform.ValueOrDie());

  for(int i: *vsi_npu_ids){
    XlaDevice::Options devopts;
    devopts.platform = platform.ValueOrDie();
    devopts.device_name_prefix = name_prefix;
    devopts.device_name = DEVICE_XLA_VSI_NPU;
    devopts.compilation_device_name = DEVICE_VSI_NPU_XLA_JIT;
    //devopts.use_multiple_streams = true;
    devopts.allowed_devices = vsi_npu_ids;
    devopts.device_ordinal = i;
    auto device = absl::make_unique<XlaDevice>(options, devopts);

    Status status = device->UseGpuDeviceInfo();
    if(!status.ok()){
      assert(true);
    }
    devices->push_back(std::move(device));
  }

  //int num_devices = p->VisibleDeviceCount();

  // for (int ordinal = 0; ordinal < num_devices; ordinal++) {
  //   devopts.device_ordinal = ordinal;
  // }

  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_VSI_NPU, XlaVsiNpuDeviceFactory, 500);

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_VSI_NPU, XlaLocalLaunchOp,
                           GetVsiNpuSupportedTypes());
REGISTER_XLA_COMPILE_KERNEL(DEVICE_XLA_VSI_NPU, XlaCompileOp,
                            GetVsiNpuSupportedTypes());
REGISTER_XLA_RUN_KERNEL(DEVICE_XLA_VSI_NPU, XlaRunOp, GetVsiNpuSupportedTypes());

REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_VSI_NPU, GetVsiNpuSupportedTypes());

REGISTER_XLA_BACKEND(DEVICE_VSI_NPU_XLA_JIT, GetVsiNpuSupportedTypes(), OpFilter);

} // tensorflow