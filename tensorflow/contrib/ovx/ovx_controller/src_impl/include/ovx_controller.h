/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef GEMM_WRAPPER_H
#define GEMM_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

//#include "ovx_nn.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define OVX_CONTROLLER_GRAPH_ID_NA  ((uint32_t)-1)

// General functions
void ovx_controller_PrintGraph(uint32_t graph_id);

int ovx_controller_GetWrapperVersion();

int ovx_controller_GetOvxBinaryVersion();

bool ovx_controller_DeInitOvx();

bool ovx_controller_InitOvx();

uint32_t ovx_controller_GetTargetGraphId();

void ovx_controller_SetTargetGraphId(uint32_t graph_id);

bool ovx_controller_FillInputNode(const char* const name, int node_id,
                                    const uint32_t * const shape, uint32_t dim_num,
                                    const uint8_t* const buf, uint64_t buf_size);

uint64_t ovx_controller_GetOutputNodeData(
    const char* node_name, uint8_t** buf, uint64_t* bytes);

// Graph functions
uint32_t ovx_controller_InstantiateGraph();

void ovx_controller_InitGraph(int version, const uint32_t graph_id);

bool ovx_controller_ConstructGraph(const uint32_t graph_id);

bool ovx_controller_ExecuteGraph(const uint32_t graph_id);

void ovx_controller_DumpNodeName(const uint32_t graph_id);

uint32_t ovx_controller_AppendNode(const char* const name, int graph_id,
                                  int node_id, int op_id,
                                  const uint8_t* const inputs,
                                  int inputs_count,
                                  const uint8_t* const outputs,
                                  int outputs_count);

bool ovx_controller_AppendConstTensor(const char* const name, int node_id,
                                       const uint32_t * const shape, uint32_t dim_num,
                                       uint8_t* data, int data_length);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // GEMM_WRAPPER_H

