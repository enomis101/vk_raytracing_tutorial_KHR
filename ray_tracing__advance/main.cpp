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


 // ImGui - standalone example application for Glfw + Vulkan, using programmable
 // pipeline If you are new to ImGui, see examples/README.txt and documentation
 // at the top of imgui.cpp.

#include <array>

#include "backends/imgui_impl_glfw.h"
#include "imgui.h"

#include "hello_vulkan.h"
#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"

#include <random>
#include <chrono>
#include <thread>
#include <ctime>  
#include <filesystem>


//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;


// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
	fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(HelloVulkan& helloVk)
{
	bool changed = false;
	auto& pc = helloVk.m_pcRaster;

	changed |= ImGuiH::CameraWidget();
	if (ImGui::CollapsingHeader("Light"))
	{
		for (int i = 0; i < NUM_LIGHTS; i++)
		{
			std::string lName = "L" + std::to_string(i);
			ImGui::Text(lName.c_str());
			Light& l = helloVk.lights[i];
			//changed |= ImGui::RadioButton("Point", &l.type, POINT_LIGHT);
			//ImGui::SameLine();
			//changed |= ImGui::RadioButton("Spot", &l.type, SPOT_LIGHT);
			//ImGui::SameLine();
			int lightType = static_cast<int>(l.type);
			changed |= ImGui::RadioButton((lName + " Infinite").c_str(), &lightType, LTYPE_INFINITE);
			ImGui::SameLine();
			changed |= ImGui::RadioButton((lName + " Area").c_str(), &lightType, LTYPE_AREA);
			l.type = static_cast<float>(lightType);

			if (l.type != LTYPE_INFINITE)
			{
				changed |= ImGui::SliderFloat3((lName + " Position").c_str(), &l.position.x, -20.f, 20.f);
			}
			//if(l.type != POINT_LIGHT && l.type != AREA_LIGHT)
			//{
			//  changed |= ImGui::SliderFloat3("Light Direction", &pc.lightDirection.x, -1.f, 1.f);
			//}
			if (l.type != LTYPE_INFINITE)
			{
				changed |= ImGui::SliderFloat((lName + " Intensity").c_str(), &l.intensity, 0.f, 1.f);
			}
			//if(l.type == SPOT_LIGHT)
			//{
			//  float dCutoff    = rad2deg(acos(pc.lightSpotCutoff));
			//  float dOutCutoff = rad2deg(acos(pc.lightSpotOuterCutoff));
			//  changed |= ImGui::SliderFloat("Cutoff", &dCutoff, 0.f, 45.f);
			//  changed |= ImGui::SliderFloat("OutCutoff", &dOutCutoff, 0.f, 45.f);
			//  dCutoff = dCutoff > dOutCutoff ? dOutCutoff : dCutoff;

			//  pc.lightSpotCutoff      = cos(deg2rad(dCutoff));
			//  pc.lightSpotOuterCutoff = cos(deg2rad(dOutCutoff));
			//}
			if (l.type == LTYPE_AREA)
			{
				changed |= ImGui::SliderFloat((lName + " Radius").c_str(), &l.radius, 1.f, 5.f);
			}

			//Add vertical spacing
			ImGui::Spacing();
		}


	}

	changed |= ImGui::SliderInt("Fr Type", &pc.fresnelType, 1, 2);
	changed |= ImGui::SliderFloat("Rough", &pc.rough, 0.f, 1.f);
	changed |= ImGui::SliderFloat("etaTD", &pc.etaTDielectric, 0.f, 5.f);
	//changed |= ImGui::SliderFloat3("etaTC", &pc.etaTConductor.x, 0.f, 5.f);
	//changed |= ImGui::SliderFloat3("kC", &pc.kConductor.x, 0.f, 5.f);

	changed |= ImGui::SliderInt("Max Frames", &helloVk.m_maxFrames, 1, 5000);
	if (changed)
		helloVk.resetFrame();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH = 1280;
static int const SAMPLE_HEIGHT = 720;


//Assuming Right Handed Coordinate System RHCS, use ctrl + shift + f on: #Assume RHCS
//to find places where this assumption is used

//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
	UNUSED(argc);

	//CWD
	//auto prova = std::filesystem::current_path();
	//std::cout << "Current working directory: " << prova << std::endl;

	// Setup GLFW window
	glfwSetErrorCallback(onErrorCallback);
	if (!glfwInit())
	{
		return 1;
	}
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);

	// Setup camera
	CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
	CameraManip.setLookat({ 8.440, 9.041, -8.973 }, { -2.462, 3.661, -0.286 }, { 0.000, 1.000, 0.000 });

	// Setup Vulkan
	if (!glfwVulkanSupported())
	{
		printf("GLFW: Vulkan Not Supported\n");
		return 1;
	}

	// setup some basic things for the sample, logging file for example
	NVPSystem system(PROJECT_NAME);

	// Search path for shaders and other media
	defaultSearchPaths = {
		NVPSystem::exePath() + PROJECT_RELDIRECTORY,
		NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
		std::string(PROJECT_NAME),
	};

	// Vulkan required extensions
	assert(glfwVulkanSupported() == 1);
	uint32_t count{ 0 };
	auto     reqExtensions = glfwGetRequiredInstanceExtensions(&count);

	// Requesting Vulkan extensions and layers
	nvvk::ContextCreateInfo contextInfo;
	contextInfo.setVersion(1, 2);                       // Using Vulkan 1.2
	for (uint32_t ext_id = 0; ext_id < count; ext_id++)  // Adding required extensions (surface, win32, linux, ..)
		contextInfo.addInstanceExtension(reqExtensions[ext_id]);
	contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);              // FPS in titlebar
	contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);  // Allow debug names
	contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);            // Enabling ability to present rendering

	// #VKRay: Activate the ray tracing extension
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
	contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);  // To build acceleration structures
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
	contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);  // To use vkCmdTraceRaysKHR
	contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
	contextInfo.addDeviceExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

	// Creating Vulkan base application
	nvvk::Context vkctx{};
	vkctx.initInstance(contextInfo);
	// Find all compatible devices
	auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
	assert(!compatibleDevices.empty());
	// Use a compatible device
	vkctx.initDevice(compatibleDevices[0], contextInfo);

	// Create example
	HelloVulkan helloVk;

	// Window need to be opened to get the surface on which to draw
	const VkSurfaceKHR surface = helloVk.getVkSurface(vkctx.m_instance, window);
	vkctx.setGCTQueueWithPresent(surface);

	helloVk.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
	helloVk.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
	helloVk.createDepthBuffer();
	helloVk.createRenderPass();
	helloVk.createFrameBuffers();

	// Setup Imgui
	helloVk.initGUI(0);  // Using sub-pass 0

	// Creation of the example
	//helloVk.loadModel(nvh::findFile("media/scenes/Medieval_building.obj", defaultSearchPaths, true));
	helloVk.loadModel(nvh::findFile("media/scenes/plane.obj", defaultSearchPaths, true));
	//helloVk.loadModel(nvh::findFile("media/scenes/wuson.obj", defaultSearchPaths, true),
	//                  nvmath::scale_mat4(nvmath::vec3f(0.5f)) * nvmath::translation_mat4(nvmath::vec3f(0.0f, 0.0f, 6.0f)));

	helloVk.loadModel(nvh::findFile("media/scenes/dragon.obj", defaultSearchPaths, true),
		nvmath::rotation_mat4_x(deg2rad(90)) * nvmath::rotation_mat4_z(deg2rad(-90)) * nvmath::scale_mat4(nvmath::vec3f(0.1f)) * nvmath::translation_mat4(nvmath::vec3f(-5.0f, -5.0f, -2.0f)));

	helloVk.loadScene();

	// helloVk.loadModel(nvh::findFile("media/scenes/sphere.obj", defaultSearchPaths, true));

	std::random_device              rd;         // Will be used to obtain a seed for the random number engine
	std::mt19937                    gen(rd());  // Standard mersenne_twister_engine seeded with rd()
	std::normal_distribution<float> dis(2.0f, 2.0f);
	std::normal_distribution<float> disn(0.5f, 0.2f);
	auto                            wusonIndex = static_cast<int>(2);

	//for(int n = 0; n < 50; ++n)
	//{
	//  ObjInstance inst;
	//  inst.objIndex       = wusonIndex;
	//  float         scale = fabsf(disn(gen));
	//  nvmath::mat4f mat   = nvmath::translation_mat4(nvmath::vec3f{dis(gen), 0.f, dis(gen) + 6});
	//  //    mat              = mat * nvmath::rotation_mat4_x(dis(gen));
	//  mat            = mat * nvmath::scale_mat4(nvmath::vec3f(scale));
	//  inst.transform = mat;

	//  helloVk.m_instances.push_back(inst);
	//}

	//// Creation of implicit geometry
	//MaterialObj mat;
	//// Reflective
	//mat.diffuse   = nvmath::vec3f(0, 0, 0);
	//mat.specular  = nvmath::vec3f(1.f);
	//mat.shininess = 0.0;
	//mat.illum     = 3;
	//helloVk.addImplMaterial(mat);
	//// Transparent
	//mat.diffuse  = nvmath::vec3f(0.4, 0.4, 1);
	//mat.illum    = 4;
	//mat.dissolve = 0.5;
	//helloVk.addImplMaterial(mat);
	//helloVk.addImplCube({-6.1, 0, -6}, {-6, 10, 6}, 0);
	//helloVk.addImplSphere({1, 2, 4}, 1.f, 1);



	helloVk.initOffscreen();
	Offscreen& offscreen = helloVk.offscreen();

	helloVk.createImplictBuffers();


	helloVk.createDescriptorSetLayout();
	helloVk.createGraphicsPipeline();
	helloVk.createUniformBuffer();
	helloVk.createObjDescriptionBuffer();
	helloVk.updateDescriptorSet();

	// #VKRay
	helloVk.initRayTracing();


	nvmath::vec4f clearColor = nvmath::vec4f(1, 1, 1, 1.00f);
	bool          useRaytracer = true;


	helloVk.setupGlfwCallbacks(window);
	ImGui_ImplGlfw_InitForVulkan(window, true);

	// Main loop
	int FPS = 60;
	while (!glfwWindowShouldClose(window))
	{
		auto start = std::chrono::system_clock::now();

		glfwPollEvents();
		if (helloVk.isMinimized())
			continue;

		// Start the Dear ImGui frame
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();


		// Show UI window.
		if (helloVk.showGui())
		{
			ImGuiH::Panel::Begin();
			bool changed = false;
			// Edit 3 floats representing a color
			changed |= ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));
			// Switch between raster and ray tracing
			changed |= ImGui::Checkbox("Ray Tracer mode", &useRaytracer);

			changed |= ImGui::SliderInt("Debug Mode", &helloVk.m_pcRaster.debug, 0, 3);
			if (changed)
				helloVk.resetFrame();

			renderUI(helloVk);
			ImGui::SliderInt("Max FPS", &FPS, 1, 120);
			ImGui::Text("Frame %d", helloVk.m_pcRaster.frame);
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGuiH::Control::Info("", "", "(F10) Toggle Pane", ImGuiH::Control::Flags::Disabled);
			ImGuiH::Panel::End();
		}

		CameraManip.updateKeyboardInput(1 / float(FPS));

		//helloVk.modifyObjTransform();

		// Start rendering the scene
		helloVk.prepareFrame();

		// Start command buffer of this frame
		auto                   curFrame = helloVk.getCurFrame();
		const VkCommandBuffer& cmdBuf = helloVk.getCommandBuffers()[curFrame];

		VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmdBuf, &beginInfo);

		// Updating camera buffer
		helloVk.updateUniformBuffer(cmdBuf);

		// Clearing screen
		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {clearColor[0], clearColor[1], clearColor[2], clearColor[3]} };
		clearValues[1].depthStencil = { 1.0f, 0 };

		// Offscreen render pass
		{
			VkRenderPassBeginInfo offscreenRenderPassBeginInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
			offscreenRenderPassBeginInfo.clearValueCount = 2;
			offscreenRenderPassBeginInfo.pClearValues = clearValues.data();
			offscreenRenderPassBeginInfo.renderPass = offscreen.renderPass();
			offscreenRenderPassBeginInfo.framebuffer = offscreen.frameBuffer();
			offscreenRenderPassBeginInfo.renderArea = { {0, 0}, helloVk.getSize() };

			// Rendering Scene
			if (useRaytracer)
			{
				helloVk.raytrace(cmdBuf, clearColor);
			}
			else
			{
				vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
				helloVk.rasterize(cmdBuf);
				vkCmdEndRenderPass(cmdBuf);
			}
		}

		// 2nd rendering pass: tone mapper, UI
		{
			VkRenderPassBeginInfo postRenderPassBeginInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
			postRenderPassBeginInfo.clearValueCount = 2;
			postRenderPassBeginInfo.pClearValues = clearValues.data();
			postRenderPassBeginInfo.renderPass = helloVk.getRenderPass();
			postRenderPassBeginInfo.framebuffer = helloVk.getFramebuffers()[curFrame];
			postRenderPassBeginInfo.renderArea = { {0, 0}, helloVk.getSize() };

			// Rendering tonemapper
			vkCmdBeginRenderPass(cmdBuf, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			offscreen.draw(cmdBuf, helloVk.getSize());

			// Rendering UI
			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
			vkCmdEndRenderPass(cmdBuf);
		}

		// Submit for display
		vkEndCommandBuffer(cmdBuf);
		helloVk.submitFrame();

		// Some computation here
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::chrono::milliseconds elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds);
		std::chrono::milliseconds x = std::chrono::milliseconds(1000 / FPS);
		std::chrono::milliseconds ms_to_sleep = x - elapsed_ms;
		std::this_thread::sleep_for(ms_to_sleep);
	}

	// Cleanup
	vkDeviceWaitIdle(helloVk.getDevice());

	helloVk.saveScene();

	helloVk.destroyResources();
	helloVk.destroy();
	vkctx.deinit();

	glfwDestroyWindow(window);
	glfwTerminate();



	return 0;
}
