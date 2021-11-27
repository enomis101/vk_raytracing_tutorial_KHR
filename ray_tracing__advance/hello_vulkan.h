/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nvvk/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "shaders/host_device.h"

// #VKRay
#include "nvvk/raytraceKHR_vk.hpp"
#include "offscreen.hpp"

#include "obj.hpp"
#include "raytrace.hpp"

#include <memory>

// Choosing the allocator to use
#define ALLOC_DMA
//#define ALLOC_DEDICATED
//#define ALLOC_VMA
#include <nvvk/resourceallocator_vk.hpp>

#if defined(ALLOC_DMA)
#include <nvvk/memallocator_dma_vk.hpp>
using Allocator = nvvk::ResourceAllocatorDma;
#elif defined(ALLOC_VMA)
#include <nvvk/memallocator_vma_vk.hpp>
using Allocator = nvvk::ResourceAllocatorVma;
#else
using Allocator = nvvk::ResourceAllocatorDedicated;
#endif


//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvk::AppBaseVk
{
public:
	HelloVulkan();
  void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void loadModel(const std::string& filename, nvmath::mat4f transform = nvmath::mat4f(1));
  void updateDescriptorSet();
  void createUniformBuffer();
  void createObjDescriptionBuffer();
  void createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures);
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterize(const VkCommandBuffer& cmdBuff);

  Offscreen& offscreen() { return m_offscreen; }
  Raytracer& raytracer() { return m_raytrace; }


  // Information pushed at each draw call
  PushConstantRaster m_pcRaster;

  // Array of objects and instances in the scene
  std::vector<ObjModel>    m_objModel;   // Model on host
  std::vector<ObjDesc>     m_objDesc;    // Model description for device access
  std::vector<ObjInstance> m_instances;  // Scene model instances


  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout;
  VkPipeline                  m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool;
  VkDescriptorSetLayout       m_descSetLayout;
  VkDescriptorSet             m_descSet;

  int  m_maxFrames{50};
  void resetFrame();
  void updateFrame();

  Light lights[NUM_LIGHTS];

  nvvk::Buffer m_bGlobals;  // Device-Host of the camera matrices
  nvvk::Buffer m_bObjDesc;  // Device buffer of the OBJ descriptions

  std::vector<nvvk::Texture> m_textures;  // vector of all textures of the scene


  Allocator       m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil m_debug;  // Utility to name objects

  // #Post
  Offscreen m_offscreen;
  void      initOffscreen();


  // #VKRay
  Raytracer m_raytrace;

  void initRayTracing();
  void raytrace(const VkCommandBuffer& cmdBuf, const nvmath::vec4f& clearColor);

  // Implicit
  ImplInst m_implObjects;

  void addImplSphere(nvmath::vec3f center, float radius, int matId);
  void addImplCube(nvmath::vec3f minumum, nvmath::vec3f maximum, int matId);
  void addImplMaterial(const MaterialObj& mat);
  void createImplictBuffers();

  void modifyObjTransform();
};

#include <memory>
#include <string>
#include <stdexcept>

template <typename... Args>
std::string string_format(const std::string& format, Args... args)
{
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
  if(size_s <= 0)
  {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  auto buf  = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

class TestSampleSphere
{
public:
  static void CoordinateSystem(const nvmath::vec3f& v1, nvmath::vec3f& v2, nvmath::vec3f& v3);
  static bool TestHandedness(const nvmath::vec3f& v1, const nvmath::vec3f& v2, const nvmath::vec3f& v3);
  static void Test();
  static std::string ToString(const nvmath::vec3f& v1) 
  {
	 std::string str;
    return string_format("[ %f, %f, %f]", v1.x, v1.y, v1.z);
  }

  static std::string ToString(const nvmath::mat3f& m)
  {
	  std::string str;
	  return string_format("| %f, %f, %f|\n| %f, %f, %f|\n| %f, %f, %f|", m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
  }

  static vec3 Reflect(const nvmath::vec3f& wo, const nvmath::vec3f& n) {
	  return -wo + 2.f * dot(wo, n) * n;
  }

  static void TestReflectVector();


#define Spectrum vec3

  // BSDF Declarations

  const uint BSDF_REFLECTION = 1 << 0;
  const uint BSDF_TRANSMISSION = 1 << 1;
  const uint BSDF_DIFFUSE = 1 << 2;
  const uint BSDF_GLOSSY = 1 << 3;
  const uint BSDF_SPECULAR = 1 << 4;
  const uint BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION |
	  BSDF_TRANSMISSION;

  const uint BXDF_LAMBERTIAN_REFLECTION = 1;

  struct BSDF
  {
	  uint reflectionType;
	  uint bxdf;
	  WaveFrontMaterial mat;
	  mat4 worldToLocal;
	  mat4 localToWorld;
  };

  struct SurfaceInteraction
  {
	  vec3 p;
	  vec3 wo;
	  vec3 n;
	  BSDF bsdf;
  };

  struct SphereAreaLight
  {
	  vec3 position;
	  mat4 transform;
	  float radius;
	  float intensity;
  };

  //UTILITY
  float absCosTheta( vec3 v)
  {
	  return std::abs(v.z);
  }

  static bool sameHemisphere( vec3 v1,  vec3 v2) {
	  return v1.z * v2.z > 0;
  }

  vec2 concentricSampleDisk( vec2 u) {
	  // Map uniform random numbers to $[-1,1]^2$
	  vec2 uOffset = 2.f * u - vec2(1, 1);

	  // Handle degeneracy at the origin
	  if (uOffset.x == 0 && uOffset.y == 0) return vec2(0, 0);

	  // Apply concentric mapping to point
	  float theta, r;
	  if (abs(uOffset.x) > abs(uOffset.y)) {
		  r = uOffset.x;
		  theta = PI_OVER4 * (uOffset.y / uOffset.x);
	  }
	  else {
		  r = uOffset.y;
		  theta = PI_OVER2 - PI_OVER4 * (uOffset.x / uOffset.y);
	  }
	  return r * vec2(cos(theta), sin(theta));
  }

  vec3 cosineSampleHemisphere( vec2 u) {
	  vec2 d = concentricSampleDisk(u);
	  float z = sqrt(std::max(0.f, 1 - d.x * d.x - d.y * d.y));
	  return vec3(d.x, d.y, z);
  }

  //BSDF FUNCTIONS

  Spectrum getBSDFValue( BSDF bsdf,  vec3 woW,  vec3 wiW)
  {
	  Spectrum f;
	  //transform woW, wiW to local Coordinates
	  vec3 wo = vec3(bsdf.worldToLocal * vec4(woW, 0.f));
	  vec3 wi = vec3(bsdf.worldToLocal * vec4(wiW, 0.f));

	  Spectrum r = bsdf.mat.diffuse;
	  f = r * INV_PI;
  }

  float getBSDFPdf( BSDF bsdf,  vec3 woW,  vec3 wiW)
  {
	  float pdf;
	  //transform woW, wiW to local Coordinates
	  vec3 wo = vec3(bsdf.worldToLocal * vec4(woW, 0.f));
	  vec3 wi = vec3(bsdf.worldToLocal * vec4(wiW, 0.f));


	  pdf = sameHemisphere(wo, wi) ? absCosTheta(wi) * INV_PI : 0;

	  return pdf;
  }

  Spectrum sampleBSDF( BSDF bsdf,  vec3 woW,  vec2 u,  vec3 wiW,  float pdf)
  {
	  Spectrum f;
	  //transform woW, wiW to local Coordinates
	  vec3 wo = vec3(bsdf.worldToLocal * vec4(woW, 0.f));
	  vec3 wi;

	// Cosine-sample the hemisphere, flipping the direction if necessary
	// to make sure wi and wo are on the same hemisphere
	wi = cosineSampleHemisphere(u);
	if (wo.z < 0)
	{
		wi.z *= -1;
	}
	pdf = getBSDFPdf(bsdf, wo, wi);
	f = getBSDFValue(bsdf, wo, wi);

	  wiW = vec3(bsdf.localToWorld * vec4(wi, 0.f));
	  return f;
  }

  static void testRandom();
  //RANDOM
  // Generate a random unsigned int from two unsigned int values, using 16 pairs
// of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"
  static uint tea(uint val0, uint val1)
  {
	  uint v0 = val0;
	  uint v1 = val1;
	  uint s0 = 0;

	  for (uint n = 0; n < 16; n++)
	  {
		  s0 += 0x9e3779b9;
		  v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		  v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	  }

	  return v0;
  }

  // Generate a random unsigned int in [0, 2^24) given the previous RNG state
  // using the Numerical Recipes linear congruential generator
  static uint lcg(uint& prev)
  {
	  uint LCG_A = 1664525u;
	  uint LCG_C = 1013904223u;
	  prev = (LCG_A * prev + LCG_C);
	  return prev & 0x00FFFFFF;
  }

  // Generate a random float in [0, 1) given the previous RNG state
  static float rnd(uint& prev)
  {
	  return (float(lcg(prev)) / float(0x01000000));
  }
};