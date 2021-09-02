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

#include "tensorflow/compiler/plugin/vsi/driver/visitors/visitor_base.h"

#include <stddef.h>

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/ops/elementwise.h"
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/reverse.h"
#include "tim/vx/ops/transpose.h"
#include "tim/vx/ops/stridedslice.h"
#include "tim/vx/ops/concat.h"
#include "tim/vx/ops/relational_operations.h"
#include "tim/vx/ops/select.h"
#include "tim/vx/ops/simple_operations.h"
#include "tim/vx/ops/reduce.h"
#include "tim/transform/layout_inference.h"
using tensorflow::str_util::StartsWith;

namespace xla {
namespace vsiplugin {

Literal BaseVisitor::evaluate(const HloComputation& computation
    /*absl::Span<const Literal* const> arg_literals*/){
    computation.Accept(this);
    return GetEvaluatedLiteralFor(computation.root_instruction()).Clone();
}

std::shared_ptr<tim::vx::Tensor> BaseVisitor::evaluate(
    const HloComputation& computation,
    std::vector<Literal>& argument_literals){
    arg_literals_ = std::move(argument_literals);
    computation.Accept(this);
    graph_->PrintGraph();
    if (!graph_->Compile()) {
        LOG(FATAL) << "Compile graph fail.";
        return nullptr;
    }
    if(!graph_->Run()){
        LOG(FATAL) << "Run graph fail";
        return nullptr;
    }
    return GetEvaluatedTensorFor(computation.root_instruction());
}

const Shape& BaseVisitor::GetOutputShape(HloInstruction* inst) const {
  return inst->shape();
}

/*the function should only be used by Handlexxxxx function so that the tensor maped to {$/hlo/$} has been created.
  the order of tensor's layout is the same as its shape: minjor to major   */
std::shared_ptr<tim::vx::Tensor> BaseVisitor::insertTranspose(const HloInstruction *hlo, std::vector<uint32_t> &dim_index){
    auto shape = hlo->shape();
    size_t dim_size = dim_index.size();
    std::vector<int64_t> output_dims(dim_size, 1);
    std::vector<uint32_t> perm(dim_size, 1);

    auto input_tensor = GetEvaluatedTensorFor(hlo);

    /*check if the shape is {WHCN} , if not, a transpose would be inserted to covert the layout. */
    bool is_need_insert_transpose = false;
    for(int i = 0; i < dim_size; i++){
        if(dim_index[i] == shape.layout().minor_to_major()[dim_size - i - 1]){
            perm[dim_size - 1 - i] = dim_size - i - 1;
        }else{
            is_need_insert_transpose = true;
            for(int j = 0; j < dim_size; j++){
                if(dim_index[i] != shape.layout().minor_to_major()[j])
                    continue;
                perm[dim_size - 1 - i] = j;
                break;
            }
        }
    }
    if(is_need_insert_transpose){
        auto input_shape = input_tensor->GetShape();
        std::vector<uint32_t> output_shape;
        for(auto d : perm){
            output_shape.push_back(input_shape[d]);
        }
        auto output_tensor = createTensorFromShape(
            convertTfPrimitiveTypeToTim(hlo->shape().element_type()), output_shape,
            tim::vx::TensorAttribute::OUTPUT);
        auto transposeOp = graph_->CreateOperation<tim::vx::ops::Transpose>(perm);
        transposeOp->BindInput(input_tensor).BindOutput(output_tensor);

        return output_tensor;
    }
    return input_tensor;
}

Status BaseVisitor::HandleElementwiseUnary(HloInstruction* hlo){
    switch (hlo->opcode()){
        case HloOpcode::kNegate:{
            LOG(INFO) << "PROCESS kNegate";
            auto shape = hlo->shape();
            const HloInstruction* input = hlo->operand(0);
            TF_RET_CHECK(ShapeUtil::SameDimensions(shape, input->shape()));
            auto inout_tensor = GetEvaluatedTensorFor(input);
            auto out_tensor = createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);

            auto neg = graph_->CreateOperation<tim::vx::ops::Neg>();
            (*neg).BindInput(inout_tensor).BindOutput(out_tensor);

            kVsiRunTensorContainer_[hlo] = out_tensor;
            break;
        }
        case HloOpcode::kExp:{
            LOG(INFO) << "PROCESS kExp";
            auto shape = hlo->shape();
            const HloInstruction* input = hlo->operand(0);
            TF_RET_CHECK(ShapeUtil::SameDimensions(shape, input->shape()));
            auto inout_tensor = GetEvaluatedTensorFor(input);
            auto out_tensor = createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);

            auto exp = graph_->CreateOperation<tim::vx::ops::Exp>();
            (*exp).BindInput(inout_tensor).BindOutput(out_tensor);

            kVsiRunTensorContainer_[hlo] = out_tensor;
            break;
        }
        case HloOpcode::kLog1p:{
            LOG(INFO) << "PROCESS kLog1p";
            auto shape = hlo->shape();
            const HloInstruction* input = hlo->operand(0);
            TF_RET_CHECK(ShapeUtil::SameDimensions(shape, input->shape()));
            auto inout_tensor = GetEvaluatedTensorFor(input);
            auto out_tensor = createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);

            auto log1p = graph_->CreateOperation<tim::vx::ops::Log>();
            (*log1p).BindInput(inout_tensor).BindOutput(out_tensor);

            kVsiRunTensorContainer_[hlo] = out_tensor;
            break;
        }
        case HloOpcode::kNot:{
            LOG(INFO) << "PROCESS kNot";
            auto shape = hlo->shape();
            const HloInstruction* input = hlo->operand(0);
            TF_RET_CHECK(ShapeUtil::SameDimensions(shape, input->shape()));
            auto inout_tensor = GetEvaluatedTensorFor(input);
            auto out_tensor = createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);

            auto logical_not = graph_->CreateOperation<tim::vx::ops::LogicalNot>();
            (*logical_not).BindInput(inout_tensor).BindOutput(out_tensor);

            kVsiRunTensorContainer_[hlo] = out_tensor;
            break;
        }
        default:
            LOG(INFO) << "not has benn implement; opcode:" << hlo->opcode();
            break;
    }
    return Status::OK();
}
Status BaseVisitor::HandleElementwiseBinary(HloInstruction* hlo){
    switch (hlo->opcode())
    {
        case HloOpcode::kAdd:{
            LOG(INFO) << "PROCESS add";
            auto shape = hlo->shape();
            const HloInstruction* lhs = hlo->operand(0);
            const HloInstruction* rhs = hlo->operand(1);
            TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
            TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

            auto lhs_tensor = GetEvaluatedTensorFor(lhs);
            auto rhs_tensor = GetEvaluatedTensorFor(rhs);
            auto out_tensor = createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);
            auto add = graph_->CreateOperation<tim::vx::ops::Add>();
            (*add).BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

            //evaluatedDevMem_[hlo] = executor_->setTensor(out_tensor);
            kVsiRunTensorContainer_[hlo] = out_tensor;
            break;
        }
        case HloOpcode::kSubtract:{
            LOG(INFO) << "PROCESS Subtract";
            auto shape = hlo->shape();
            const HloInstruction* lhs = hlo->operand(0);
            const HloInstruction* rhs = hlo->operand(1);
            TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
            TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

            auto lhs_tensor = GetEvaluatedTensorFor(lhs);
            auto rhs_tensor = GetEvaluatedTensorFor(rhs);
            auto out_tensor = createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);
            auto sub = graph_->CreateOperation<tim::vx::ops::Sub>();
            (*sub).BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

            //evaluatedDevMem_[hlo] = executor_->setTensor(out_tensor);
            kVsiRunTensorContainer_[hlo] = out_tensor;
            break;
        }
        case HloOpcode::kMultiply:{
            LOG(INFO) << "PROCESS Multiply";
            auto shape = hlo->shape();
            const HloInstruction* lhs = hlo->operand(0);
            const HloInstruction* rhs = hlo->operand(1);
            TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
            TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

            auto lhs_tensor = GetEvaluatedTensorFor(lhs);
            auto rhs_tensor = GetEvaluatedTensorFor(rhs);
            auto out_tensor = createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);
            auto mul = graph_->CreateOperation<tim::vx::ops::Multiply>();
            (*mul).BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

            //evaluatedDevMem_[hlo] = executor_->setTensor(out_tensor);
            kVsiRunTensorContainer_[hlo] = out_tensor;
            break;
        }
        case HloOpcode::kDivide:{
            LOG(INFO) << "PROCESS kDivide";
            auto shape = hlo->shape();
            const HloInstruction* lhs = hlo->operand(0);
            const HloInstruction* rhs = hlo->operand(1);
            TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));

            auto lhs_tensor = GetEvaluatedTensorFor(lhs);
            auto rhs_tensor = GetEvaluatedTensorFor(rhs);
            auto out_tensor = createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);
            auto div = graph_->CreateOperation<tim::vx::ops::Div>();
            if(rhs_tensor->IsConstTensor() || rhs_tensor->GetDataType() == tim::vx::DataType::INT32){
                auto timShape = rhs_tensor->GetShape();
                auto dataType = rhs_tensor->GetDataType();
                auto spec = rhs_tensor->GetSpec();

                tim::vx::TensorSpec timSpec(tim::vx::DataType::FLOAT32, timShape,
                    tim::vx::TensorAttribute::INPUT);

                uint32_t size = 1;
                for(uint32_t i=0;i<timShape.size();i++){
                    size = size*timShape[i];
                }
                size = size*4;

                int32_t buffer_data;
                rhs_tensor->CopyDataFromTensor(static_cast<void*>(&buffer_data));
                float buffer_data_transform = (float)buffer_data;

                auto rhs_tensor_transform = graph_->CreateTensor(timSpec,static_cast<void*>(&buffer_data_transform));
                (*div).BindInput(lhs_tensor).BindInput(rhs_tensor_transform).BindOutput(out_tensor);

            }else{
                (*div).BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);
            }
            kVsiRunTensorContainer_[hlo] = out_tensor;
            break;
        }
        default:
            LOG(INFO) << "not has benn implement; opcode:" << hlo->opcode();
            break;
    }
    return Status::OK();
}

Status BaseVisitor::FinishVisit(HloInstruction* root){
    return Status::OK();
}

Status BaseVisitor::Unimplemented(HloInstruction* inst) {
  return xla::Unimplemented("%s (%s) not implemented", inst->name().c_str(),
                            HloOpcodeString(inst->opcode()).c_str());
}

Status BaseVisitor::HandleConvert(HloInstruction* hlo){
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    const HloInstruction* input = hlo->operand(0);
    auto it = kVsiRunTensorContainer_.find(input);
    kVsiRunTensorContainer_[hlo] = it->second;
    return Status::OK();
}

// VSI TODO::it is just hack, need to implement
Status BaseVisitor::HandleGetTupleElement(HloInstruction* hlo){
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    const HloInstruction* input = hlo->operand(0);
    auto it = kVsiRunTensorContainer_.find(input);
    kVsiRunTensorContainer_[hlo] = it->second;
    return Status::OK();
}

Status BaseVisitor::HandleReduce(HloInstruction* hlo) {
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    auto shape = hlo->shape();
    auto* reduce_hlo = Cast<HloReduceInstruction>(hlo);

    CHECK_EQ(reduce_hlo->input_count(), 1);

    const HloInstruction* input = reduce_hlo->operand(0);
    auto input_tensor = GetEvaluatedTensorFor(input);

    auto dimensions = hlo->dimensions();
    std::cout<<"HandleReduce dimension "<<dimensions[0]<<std::endl;

    std::vector<int32_t> axis;
    for(uint32_t i=0;i<dimensions.size();i++){
        std::cout<<dimensions[i]<<std::endl;
        axis.push_back(static_cast<int32_t>(dimensions[i]));
    }

    auto out_tensor =
      createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);

    auto reduce = graph_->CreateOperation<tim::vx::ops::ReduceMean>(axis,false);
    (*reduce).BindInput(input_tensor).BindOutput(out_tensor);
    kVsiRunTensorContainer_[hlo] = out_tensor;
    return Status::OK();
}


Status BaseVisitor::HandleCompare(HloInstruction* hlo) {
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    auto shape = hlo->shape();
    const HloInstruction* lhs = hlo->operand(0);
    const HloInstruction* rhs = hlo->operand(1);
    TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
    TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

    auto lhs_tensor = GetEvaluatedTensorFor(lhs);
    auto rhs_tensor = GetEvaluatedTensorFor(rhs);
    auto out_tensor =
      createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);
    auto compare = graph_->CreateOperation<tim::vx::ops::Equal>();
    (*compare).BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

    kVsiRunTensorContainer_[hlo] = out_tensor;
    return Status::OK();
}

Status BaseVisitor::HandleSelect(HloInstruction* hlo) {
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    auto shape = hlo->shape();
    const HloInstruction* condition = hlo->operand(0);
    const HloInstruction* lhs = hlo->operand(1);
    const HloInstruction* rhs = hlo->operand(2);
    TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
    TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

    auto condition_tensor = GetEvaluatedTensorFor(condition);
    auto lhs_tensor = GetEvaluatedTensorFor(lhs);
    auto rhs_tensor = GetEvaluatedTensorFor(rhs);
    auto out_tensor =
      createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);
    auto select = graph_->CreateOperation<tim::vx::ops::Select>();
    (*select).BindInput(condition_tensor).BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

    kVsiRunTensorContainer_[hlo] = out_tensor;
    return Status::OK();
}

// VSI TODO::it is just hack, need to implement
Status BaseVisitor::HandleTuple(HloInstruction* hlo){
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    const HloInstruction* input = hlo->operand(0);
    auto it = kVsiRunTensorContainer_.find(input);
    kVsiRunTensorContainer_[hlo] = it->second;
    return Status::OK();
}

Status BaseVisitor::HandleBroadcast(HloInstruction* hlo){
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    const HloInstruction* input = hlo->operand(0);
    auto it = kVsiRunTensorContainer_.find(input);
    kVsiRunTensorContainer_[hlo] = it->second;
    return Status::OK();
}

Status BaseVisitor::HandleConcatenate(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  const HloInstruction* lhs = hlo->operand(0);
  const HloInstruction* rhs = hlo->operand(1);
  //   TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
  //   TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

  auto dims = hlo->dimensions();

  std::cout<<std::endl;
  auto lhs_tensor = GetEvaluatedTensorFor(lhs);
  auto rhs_tensor = GetEvaluatedTensorFor(rhs);
  auto out_tensor =
      createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);
  auto concat = graph_->CreateOperation<tim::vx::ops::Concat>(dims[0],2);
  (*concat).BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

  kVsiRunTensorContainer_[hlo] = out_tensor;
  return Status::OK();
}

Status BaseVisitor::HandleTranspose(HloInstruction* transpose){
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    const HloInstruction* input = transpose->operand(0);
    const Shape& in_shape = input->shape();
    const Shape& out_shape = transpose->shape();

    TF_CHECK_OK(ShapeUtil::ValidateShape(in_shape));
    TF_CHECK_OK(ShapeUtil::ValidateShape(out_shape));
    CHECK(ShapeUtil::SameElementType(in_shape, out_shape));
    CHECK_EQ(transpose->dimensions().size(), in_shape.rank());
    CHECK_EQ(in_shape.rank(), out_shape.rank());

    auto in_tensor = GetEvaluatedTensorFor(input);
    auto out_tensor = createTensorFromShape(out_shape, tim::vx::TensorAttribute::OUTPUT);

    std::vector<uint32_t> tmpdims;
    auto input_minor_to_major = input->shape().layout().minor_to_major();

    for(auto d: input_minor_to_major){
        tmpdims.push_back(transpose->dimensions(d));
    }

    std::vector<uint32_t> dims;
    for(auto d: tmpdims){
        uint32 i = 0;
        for(i = 0; i < input_minor_to_major.size(); i++){
            if(input_minor_to_major[i] == d)
            break;
        }
        dims.push_back(i);
    }

    auto transposeOp = graph_->CreateOperation<tim::vx::ops::Transpose>(dims);
    transposeOp->BindInput(in_tensor).BindOutput(out_tensor);

    //evaluatedDevMem_[transpose] = executor_->setTensor(out_tensor);
    kVsiRunTensorContainer_[transpose] = out_tensor;
    return Status::OK();
}

Status BaseVisitor::HandleReverse(HloInstruction* hlo){
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    const HloInstruction* input = hlo->operand(0);
    const Shape& in_shape = input->shape();
    const Shape& out_shape = hlo->shape();

    TF_CHECK_OK(ShapeUtil::ValidateShape(in_shape));
    TF_CHECK_OK(ShapeUtil::ValidateShape(out_shape));
    CHECK(ShapeUtil::SameElementType(in_shape, out_shape));
    // CHECK_EQ(hlo->dimensions().size(), in_shape.rank());
    CHECK_EQ(in_shape.rank(), out_shape.rank());

    auto in_tensor = GetEvaluatedTensorFor(input);
    auto out_tensor = createTensorFromShape(out_shape, tim::vx::TensorAttribute::OUTPUT);

    std::vector<uint32_t> tmpdims;
    auto input_minor_to_major = input->shape().layout().minor_to_major();

    for(auto d: input_minor_to_major){
        tmpdims.push_back(hlo->dimensions(d));
    }

    std::vector<int32_t> dims;
    for(auto d: tmpdims){
        uint32 i = 0;
        for(i = 0; i < input_minor_to_major.size(); i++){
            if(input_minor_to_major[i] == d) {
                dims.push_back(i);
            }
        }
    }

    // auto dims0 = hlo->dimensions();
    // std::vector<int32_t> dims = convert_array<std::vector<int32_t>>(dims0);
    auto reverseOp = graph_->CreateOperation<tim::vx::ops::Reverse>(dims);
    reverseOp->BindInput(in_tensor).BindOutput(out_tensor);

    //evaluatedDevMem_[transpose] = executor_->setTensor(out_tensor);
    kVsiRunTensorContainer_[hlo] = out_tensor;
    return Status::OK();
}

Status BaseVisitor::HandleReshape(HloInstruction* hlo){
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    auto shape = hlo->shape();
    const HloInstruction* input = hlo->operand(0);
    auto in_tensor = GetEvaluatedTensorFor(input);
    auto out_tensor = createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);

    std::vector<uint32_t> dims;
    for(auto d: shape.dimensions()){
        dims.push_back(d);
    }
    auto reshapeOp = graph_->CreateOperation<tim::vx::ops::Reshape>(dims);
    (*reshapeOp).BindInput(in_tensor).BindOutput(out_tensor);

    //evaluatedDevMem_[hlo] = executor_->setTensor(out_tensor);
    kVsiRunTensorContainer_[hlo] = out_tensor;
    return Status::OK();
}

Status BaseVisitor::HandleParameter(HloInstruction* hlo){
    CHECK_LT(hlo->parameter_number(), arg_literals_.size());
    LOG(INFO) << "PROCESS " << __FUNCTION__;
    auto& input_literal = arg_literals_[hlo->parameter_number()];
    VLOG(2) << "Parameter evaluated to: " << input_literal.ToString();
    DCHECK(Shape::Equal().MinorToMajorOnlyInLayout()(hlo->shape(),
                                                    input_literal.shape()))
        << "parameter shape is: "
        << ShapeUtil::HumanStringWithLayout(hlo->shape())
        << ", but input literal shape is: "
        << ShapeUtil::HumanStringWithLayout(input_literal.shape());

        if(kVsiRunTensorContainer_.find(hlo) == kVsiRunTensorContainer_.end()){
            ShapeIndex shapeIndex({});
            void *buffer = input_literal.untyped_data(shapeIndex);
            auto timTensor = createTensorFromShape(input_literal.shape());
            timTensor->CopyDataToTensor(buffer);
            kVsiRunTensorContainer_[hlo] = timTensor;
        }

    return Status::OK();
}

Status BaseVisitor::HandleConstant(HloInstruction* hlo){
    LOG(INFO) << "PROCESS Constant";

    if(kVsiRunTensorContainer_.find(hlo) == kVsiRunTensorContainer_.end()){
        ShapeIndex shapeIndex({});

        auto& literal = hlo->literal();
        const void *buffer = literal.untyped_data(shapeIndex);
        auto timTensor = createTensorFromShape(literal.shape());
        timTensor->CopyDataToTensor(buffer);
        kVsiRunTensorContainer_[hlo] = timTensor;
    }

    return Status::OK();
}

Status BaseVisitor::HandleConvolution(HloInstruction* conv) {
    LOG(INFO) << "PROCESS " << __FUNCTION__;

    auto lhs = conv->operand(0);
    auto rhs = conv->operand(1);
    const auto& window = conv->window();
    const Shape& result_shape = conv->shape();
    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();

    TF_CHECK_OK(ShapeUtil::ValidateShape(lhs_shape));
    TF_CHECK_OK(ShapeUtil::ValidateShape(rhs_shape));
    CHECK(lhs_shape.IsArray());
    CHECK(rhs_shape.IsArray());
    CHECK(ShapeUtil::SameElementType(lhs_shape, rhs_shape));
    CHECK(ShapeUtil::SameElementType(lhs_shape, result_shape));

    const auto& dnums = conv->convolution_dimension_numbers();
    const int64 num_spatial_dims = dnums.output_spatial_dimensions_size();
    CHECK_EQ(num_spatial_dims, dnums.input_spatial_dimensions_size());
    CHECK_EQ(num_spatial_dims, dnums.kernel_spatial_dimensions_size());
    CHECK_EQ(num_spatial_dims, 2); /*vsi requirement*/
    CHECK_GE(num_spatial_dims, 0);
    CHECK_EQ(window.dimensions_size(), num_spatial_dims);

    const auto lhs_rank = lhs_shape.rank();
    const auto rhs_rank = rhs_shape.rank();
    CHECK_EQ(num_spatial_dims + 2, lhs_rank);
    CHECK_EQ(num_spatial_dims + 2, rhs_rank);

    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferConvolveShape(
                            lhs_shape, rhs_shape, conv->feature_group_count(),
                            conv->batch_group_count(), window, dnums));
    CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape set to: " << ShapeUtil::HumanString(result_shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    // prepare parameter for vsi.
    std::vector<uint32_t> input_dim;
    input_dim.push_back(dnums.input_batch_dimension());
    input_dim.push_back(dnums.input_feature_dimension());

    std::vector<uint32_t> weight_dim;
    weight_dim.push_back(dnums.kernel_output_feature_dimension());
    weight_dim.push_back(dnums.kernel_input_feature_dimension());
    for(size_t i = 2; i < lhs_rank; i++){
        input_dim.push_back(dnums.input_spatial_dimensions(i - 2));
        weight_dim.push_back(dnums.kernel_spatial_dimensions(i - 2));
    }

    /*prepare input and weight that whose shape is WHCN, layout minor to major:{0,1,2,3}*/
    auto input = insertTranspose(lhs, input_dim);
    auto weight = insertTranspose(rhs, weight_dim);

    std::array<uint32_t, 2> ksize = {window.dimensions(1).size(), window.dimensions(0).size()};
    std::array<uint32_t, 2> stride = {window.dimensions(1).stride(), window.dimensions(0).stride()};
    std::array<uint32_t, 2> dilation = {window.dimensions(1).window_dilation(), window.dimensions(0).window_dilation()};
    std::array<uint32_t, 4> pad = {window.dimensions(1).padding_low(), window.dimensions(1).padding_high(),
            window.dimensions(0).padding_low(), window.dimensions(0).padding_high()};
    auto convOp = graph_->CreateOperation<tim::vx::ops::Conv2d>(dnums.kernel_output_feature_dimension(),
                                tim::vx::PadType::AUTO, ksize, stride, dilation, pad);

    auto out_tensor = createTensorFromShape(conv->shape(), tim::vx::TensorAttribute::OUTPUT);
    convOp->BindInput(input).BindInput(weight).BindOutput(out_tensor);

    kVsiRunTensorContainer_[conv] = out_tensor;
    return Status::OK();
}
}  // namespace vsiplugin
}  // namespace xla