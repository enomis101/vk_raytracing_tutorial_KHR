/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef COMMON_HOST_DEVICE
#define COMMON_HOST_DEVICE

#ifdef __cplusplus
#include "nvmath/nvmath.h"
// GLSL Type
using vec2 = nvmath::vec2f;
using vec3 = nvmath::vec3f;
using vec4 = nvmath::vec4f;
using mat4 = nvmath::mat4f;
using uint = unsigned int;
#endif

#define PI 3.14159265358979323846f
#define INV_PI 0.31830988618379067154f
#define INV_2PI  0.15915494309189533577
#define INV_4PI  0.07957747154594766788
#define PI_OVER2 1.57079632679489661923
#define PI_OVER4 0.78539816339744830961

// clang-format off
#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
 #define START_BINDING(a) enum a {
 #define END_BINDING() }
#else
 #define START_BINDING(a)  const uint
 #define END_BINDING() 
#endif

#define DEBUG_RGEN 1
#define DEBUG_RCHIT 2

START_BINDING(SceneBindings)
  eGlobals   = 0,  // Global uniform containing camera matrices
  eObjDescs  = 1,  // Access to the object descriptions
  eTextures  = 2,  // Access to textures
  eImplicits = 3   // Implicit objects
END_BINDING();

START_BINDING(RtxBindings)
  eTlas     = 0,  // Top-level acceleration structure
  eOutImage = 1   // Ray tracer output image
END_BINDING();
// clang-format on

// Information of a obj model when referenced in a shader
struct ObjDesc
{
  int      txtOffset;             // Texture index offset in the array of textures
  uint64_t vertexAddress;         // Address of the Vertex buffer
  uint64_t indexAddress;          // Address of the index buffer
  uint64_t materialAddress;       // Address of the material buffer
  uint64_t materialIndexAddress;  // Address of the triangle material index buffer
};

#define NUM_LIGHTS 2

#define LTYPE_AREA 1
#define LTYPE_INFINITE 2

struct Light
{
	vec4 position;
	vec4 direction;

	float intensity;
	float radius;

	float   type;
	float padding;
};

// Uniform buffer set at each frame
struct GlobalUniforms
{
  mat4 viewProj;     // Camera view * projection
  mat4 viewInverse;  // Camera inverse view matrix
  mat4 projInverse;  // Camera inverse projection matrix

  Light lights[NUM_LIGHTS];
};

//// Push constant structure for the raster
//struct PushConstantRaster
//{
//  mat4  modelMatrix;  // matrix of the instance
//  uint  objIndex;
//
//  int   frame;
//  int debug;
//
//  //float rough;
//  //int fresnelType;
//  //float etaTDielectric;
//  //vec3 etaTConductor;
//  //vec3 kConductor;
//};

// Push constant structure for the raster
struct PushConstantRaster
{
	mat4  modelMatrix;  // matrix of the instance

	uint  objIndex;
	int   frame;
	int debug;

	int fresnelType;

	float rough;
	float etaTDielectric;

	
	//vec3 etaTConductor;
	//vec3 kConductor;
};


//// Push constant structure for the ray tracer
//struct PushConstantRay
//{
//  vec4  clearColor;
//  uint  objIndex;
//
//  int   frame;
//  int debug;
//
//  //float rough;
//  //int fresnelType;
//  //float etaTDielectric;
//  //vec3 etaTConductor;
//  //vec3 kConductor;
//};

struct PushConstantRay
{
	vec4  clearColor;
	vec4  clearColor2;
	vec4  clearColor3;
	vec4  clearColor4;
	uint  objIndex;
	int   frame;
	int debug;

	int fresnelType;

	float rough;
	float etaTDielectric;
	
	//vec3 etaTConductor;
	//vec3 kConductor;
};

struct Vertex  // See ObjLoader, copy of VertexObj, could be compressed for device
{
  vec3 pos;
  vec3 nrm;
  vec3 color;
  vec2 texCoord;
};

struct WaveFrontMaterial  // See ObjLoader, copy of MaterialObj, could be compressed for device
{
  vec3  ambient;
  vec3  diffuse;
  vec3  specular;
  vec3  transmittance;
  vec3  emission;
  float shininess;
  float ior;       // index of refraction
  float dissolve;  // 1 == opaque; 0 == fully transparent
  int   illum;     // illumination model (see http://www.fileformat.info/format/material/)
  int   textureId;
  int bxdf;
};


#endif
