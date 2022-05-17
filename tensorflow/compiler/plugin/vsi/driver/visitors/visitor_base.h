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

#ifndef TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VISITORS_VISITOR_BASE_H_
#define TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VISITORS_VISITOR_BASE_H_

#include <mutex>
#include <string>
#include <unordered_map>

#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
namespace xla {
namespace vsiplugin {

/*
 * The base visitor handles all operations that are element-wise.
 * This includes all explicitly element-wise ops, for temprarily, they
 * are implemented by hlo_evaluator, and repalce it with AIM implment
 * step by step. All of these have no element to element dependencies.
 */
class BaseVisitor : public DfsHloVisitor {
 public:
  BaseVisitor(VsiExecutor* executor)
      : executor_(executor), graph_(executor->getContext()->CreateGraph()){};

  std::shared_ptr<tim::vx::Tensor> createTensorFromTupleShape(
      const Shape& shape, int64 index,
      tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::INPUT);

  std::shared_ptr<tim::vx::Tensor> createTensorFromShape(
      const Shape& shape,
      tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::INPUT);

  std::shared_ptr<tim::vx::Tensor> createTensorFromShape(
      tim::vx::DataType dataType, std::vector<uint32_t> shape,
      tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::INPUT);

  static tim::vx::DataType convertTfPrimitiveTypeToTim(
      xla::PrimitiveType xlaType);

  /*dim_index: store the demension index info of the $hlo$ as order
    major_to_minor: {N, C, ..... } if it should be inserted a transpose, its
    output would be returned.*/
  std::shared_ptr<tim::vx::Tensor> insertTranspose(
      const HloInstruction* hlo, std::vector<uint32_t>& dim_index);

  virtual const Shape& GetOutputShape(HloInstruction*) const;

  Literal evaluate(const HloComputation& computation
                   /*absl::Span<const Literal* const> arg_literals*/);

  std::vector<std::shared_ptr<tim::vx::Tensor>> evaluate(
      const HloComputation& computation,
      std::vector<Literal>& argument_literals);

  Status HandleHloOp(HloInstruction* hlo);

  Status FinishVisit(HloInstruction* root) final;

  // Returns the already-evaluated literal result for the instruction.
  //
  // A Constant instruction is considered evaluated and its literal will be
  // returned directly without looking up the cache.
  //
  // Similarly, a Parameter instruction is considered evaluated and its literal
  // is looked up in arg_literals.
  //
  // Crash with log if the given instruction has not been evaluated previously.
  const Literal& GetEvaluatedLiteralFor(const HloInstruction* hlo) {
    if (hlo->IsConstant()) {
      return hlo->literal();
    }
    // if (hlo->opcode() == HloOpcode::kParameter) {
    //     return *arg_literals_.at(hlo->parameter_number());
    // }
    auto it = evaluated_.find(hlo);
    CHECK(it != evaluated_.end())
        << "could not find evaluated value for: " << hlo->ToString();
    return it->second;
  }

  const std::vector<std::shared_ptr<tim::vx::Tensor>> GetEvaluatedTensorFor(
      const HloInstruction* hlo) {
    // return createTensorFromShape(hlo->shape());
    auto it = kVsiRunTensorContainer_.find(hlo);
    CHECK(it != kVsiRunTensorContainer_.end())
        << "could not find evaluated value for: " << hlo->ToString();
    return kVsiRunTensorContainer_[hlo];
  }

  // Called by HandleElementwiseBinarythe FinishVisit.
  virtual Status FinishScopedVisit(HloInstruction* root) {
    return Status::OK();
  }

  template <typename T>
  Status HandleSimpleElementwiseBinary(HloInstruction* hlo);

  template <typename T>
  Status HandleSimpleElementwiseUnary(HloInstruction* hlo);

  template <typename T>
  Status CreateCompareOp(std::shared_ptr<tim::vx::Tensor>& lhs_tensor,
                         std::shared_ptr<tim::vx::Tensor>& rhs_tensor,
                         std::shared_ptr<tim::vx::Tensor>& out_tensor);

  template <typename T>
  Status CreateReduceOp(std::shared_ptr<tim::vx::Tensor>& input,
                        std::shared_ptr<tim::vx::Tensor>& output,
                        std::vector<int32_t>& axis);

  Status HandleReduceOpMap(HloOpcode opcode,
                           std::shared_ptr<tim::vx::Tensor>& input,
                           std::shared_ptr<tim::vx::Tensor>& output,
                           std::vector<int32_t>& axis);

  Status HandleElementwiseBinary(HloInstruction* hlo) override;

  Status HandleElementwiseUnary(HloInstruction* hlo) override;

  Status HandleConstant(HloInstruction* hlo) override;

  Status HandleParameter(HloInstruction* hlo) override;

  Status HandleReshape(HloInstruction* hlo) override;

  Status HandleTranspose(HloInstruction* hlo) override;

  Status HandleTuple(HloInstruction* hlo) override;

  Status HandleGetTupleElement(HloInstruction* hlo) override;

  Status HandleConvolution(HloInstruction* hlo) override;

  Status HandleReverse(HloInstruction* hlo) override;

  Status HandleConvert(HloInstruction* hlo) override;

  // Status HandleSlice(HloInstruction* hlo) override;

  Status HandleBroadcast(HloInstruction* hlo) override;

  Status HandleConcatenate(HloInstruction* hlo) override;

  Status HandleCompare(HloInstruction* hlo) override;

  Status HandleSelect(HloInstruction* hlo) override;

  Status HandleReduce(HloInstruction* hlo) override;

  Status HandleReduceWindow(HloInstruction* hlo) override;

  Status HandleDot(HloInstruction* hlo) override;

  Status HandleIota(HloInstruction* hlo) override;

  Status HandleCopy(HloInstruction* hlo) override;

#define HANDLE_AS_HLO_OP(Name) \
  Status Name(HloInstruction* inst) override { return HandleHloOp(inst); }

  /*
   * Operations not processed by this visitor.
   */
#define UNIMPLEMENTED(Name)                                     \
  Status Name(HloInstruction* inst) override {                  \
    LOG(INFO) << "@@ unimplement instruction " << __FUNCTION__; \
    return Unimplemented(inst);                                 \
  };

  UNIMPLEMENTED(HandleTupleSelect)
  // UNIMPLEMENTED(HandleConvert)
  UNIMPLEMENTED(HandleCollectivePermuteStart)
  UNIMPLEMENTED(HandleCollectivePermuteDone)
  UNIMPLEMENTED(HandleRngBitGenerator)
  UNIMPLEMENTED(HandleBitcastConvert)
  UNIMPLEMENTED(HandleAllReduce)
  UNIMPLEMENTED(HandleAllGather)
  UNIMPLEMENTED(HandleFusion)
  UNIMPLEMENTED(HandleCall)
  UNIMPLEMENTED(HandleCustomCall)
  UNIMPLEMENTED(HandleMap)
  UNIMPLEMENTED(HandleConditional)
  UNIMPLEMENTED(HandleInfeed)
  UNIMPLEMENTED(HandleAfterAll)
  UNIMPLEMENTED(HandleReal)
  UNIMPLEMENTED(HandleAllToAll)
  UNIMPLEMENTED(HandleAddDependency)
  // UNIMPLEMENTED(HandleElementwiseUnary)
  UNIMPLEMENTED(HandleClamp)
  // UNIMPLEMENTED(HandleSelect)
  // UNIMPLEMENTED(HandleCompare)
  UNIMPLEMENTED(HandleRng)
  UNIMPLEMENTED(HandleSlice)
  UNIMPLEMENTED(HandleDynamicSlice)
  UNIMPLEMENTED(HandleDynamicUpdateSlice)
  UNIMPLEMENTED(HandleSelectAndScatter)
  UNIMPLEMENTED(HandleWhile)
  UNIMPLEMENTED(HandlePad)
  UNIMPLEMENTED(HandleSort)
  // UNIMPLEMENTED(HandleReduce)
  UNIMPLEMENTED(HandleBitcast)
  // UNIMPLEMENTED(HandleBroadcast)
  UNIMPLEMENTED(HandleReducePrecision)
  UNIMPLEMENTED(HandleOutfeed)
  UNIMPLEMENTED(HandleSend)
  UNIMPLEMENTED(HandleSendDone)
  UNIMPLEMENTED(HandleRecv)
  UNIMPLEMENTED(HandleRecvDone)
  UNIMPLEMENTED(HandleBatchNormInference)
  UNIMPLEMENTED(HandleBatchNormTraining)
  UNIMPLEMENTED(HandleBatchNormGrad)
  UNIMPLEMENTED(HandleFft)
  UNIMPLEMENTED(HandleGather)
  // UNIMPLEMENTED(HandleCopy)
  // UNIMPLEMENTED(HandleIota)
  UNIMPLEMENTED(HandleScatter)
  UNIMPLEMENTED(HandleCollectivePermute)
  // UNIMPLEMENTED(HandleConcatenate)
  UNIMPLEMENTED(HandleGetDimensionSize)
  UNIMPLEMENTED(HandleReplicaId)
  UNIMPLEMENTED(HandleTriangularSolve)
  UNIMPLEMENTED(HandleCholesky)
  UNIMPLEMENTED(HandlePartitionId)
  UNIMPLEMENTED(HandleRngGetAndUpdateState)
  UNIMPLEMENTED(HandleCopyStart)
  UNIMPLEMENTED(HandleCopyDone)
  UNIMPLEMENTED(HandleSetDimensionSize)
  // UNIMPLEMENTED(HandleDot)
  // UNIMPLEMENTED(HandleReduceWindow)
  UNIMPLEMENTED(HandleDynamicReshape)
  UNIMPLEMENTED(HandleAllGatherStart)
  UNIMPLEMENTED(HandleAllGatherDone)
  UNIMPLEMENTED(HandleReduceScatter)
  UNIMPLEMENTED(HandleAllReduceStart)
  UNIMPLEMENTED(HandleAllReduceDone)

 protected:
  Status Unimplemented(HloInstruction* inst);

  const std::string name_ = "vsi base visitor";

  std::unique_ptr<HloEvaluator> cpu_evaluator_;

 private:
  VsiExecutor* executor_;

  // Tracks the HLO instruction and its evaluated literal result.
  // Parameters and constants aren't stored here,
  // TODO: it is better the Literal value was repalced with device memory
  //       handle.
  std::mutex mutex_;
  std::unordered_map<const HloInstruction*, Literal> evaluated_
      TF_GUARDED_BY(mutex_);
  std::unordered_map<const HloInstruction*,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>>
      kVsiRunTensorContainer_ TF_GUARDED_BY(mutex_);
  // std::unordered_map<const HloInstruction*, std::shared_ptr<tim::vx::Tensor>>
  //     kVsiRunTensorContainer_ TF_GUARDED_BY(mutex_);
  std::vector<Literal> arg_literals_;
  std::unordered_map<int64, uint32_t> kVsiInputId_ TF_GUARDED_BY(mutex_);
  std::shared_ptr<tim::vx::Graph> graph_;
#if THRIFT_RPC
 public:
  std::shared_ptr<tim::vx::platform::IExecutable> remote_exectable_ = nullptr;
  std::vector<std::shared_ptr<tim::vx::platform::ITensorHandle>> remote_outputs_;
#endif
};

}  // namespace vsiplugin
}  // namespace xla

#endif
