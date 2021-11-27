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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_debug_printf : enable

#include "raycommon.glsl"
#include "wavefront.glsl"
#include "random.glsl"
#include "sample.glsl"
#include "bsdf.glsl"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(buffer_reference, scalar) buffer Vertices {Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices {ivec3 i[]; }; // Triangle indices
layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MatIndices {int i[]; }; // Material ID for each triangle
layout(set = 0, binding = eTlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = eObjDescs, scalar) buffer ObjDesc_ { ObjDesc i[]; } objDesc;
layout(set = 1, binding = eTextures) uniform sampler2D textureSamplers[];
layout(set = 1, binding = eGlobals) uniform _GlobalUniforms { GlobalUniforms uni; };

layout(push_constant) uniform _PushConstantRay { PushConstantRay pcRay; };
// clang-format on


layout(location = 3) callableDataEXT rayLight cLight;

Spectrum estimateDirect(in SurfaceInteraction si, in Light light, inout uint seed)
{
    Spectrum Ld = Spectrum(0.f);
    // Sample light source with multiple importance sampling

    float lightPdf = 0, scatteringPdf = 0;
	vec2 uLight = vec2(rnd(seed), rnd(seed));
	vec3 samplePoint;
	vec3 sampleNormal;
	Spectrum Li;

	if(sampleLight(light, si.p, uLight, samplePoint, sampleNormal, lightPdf))
	{
		Li = Spectrum(light.intensity);
	}
	else
	{
		Li = vec3(0.f);
		return Li;
	}
	vec3 wi = normalize(samplePoint - si.p);

	//call sampleSphere
	if (lightPdf > 0 && Li != Spectrum(0.f))
	{
		Spectrum f;

		f = BSDF_f(si.bsdf, si.wo, wi) * abs(dot(wi, si.n));
		scatteringPdf = BSDF_Pdf(si.bsdf, si.wo, wi);

		//check if the hitPoint and the sampled point are not occluded
		//by spawning a ray from hitPoint to lightSamplePoint
		if(f != Spectrum(0.f))
		{
			// cast shadow ray 
			float tMin   = 0.001;
			float tMax   = distance(samplePoint, si.p) + light.radius * 2;
			//vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
			vec3  origin =  si.p + wi * tMin;
			vec3  rayDir = wi;
			uint  flags  = gl_RayFlagsSkipClosestHitShaderEXT;
			isShadowed   = true;
			traceRayEXT(topLevelAS,  // acceleration structure
						flags,       // rayFlags
						0xFF,        // cullMask
						0,           // sbtRecordOffset
						0,           // sbtRecordStride
						1,           // missIndex
						origin,      // ray origin
						tMin,        // ray min range
						rayDir,      // ray direction
						tMax,        // ray max range
						1            // payload (location = 1)
			);
			if(isShadowed)
			{
				Li = Spectrum(0.f);
			}
			if(Li != Spectrum(0.f))
			{
				float weight = powerHeuristic(1, lightPdf, 1, scatteringPdf);
                Ld += f * Li * weight / lightPdf;
			}
		}
	}

	// Sample BSDF with multiple importance sampling
	// In practice compute a new direction at this hitPoint
	// and calculate the weight of the light 

	//We do not check if delta light because we assume for now only one area light
    Spectrum f;
	//sampledSpecular for now will be always false because the bsdf 
	// contains just one BxDF
    bool sampledSpecular = false;
   // assume always surface interacton

    // Sample scattered direction for surface interactions
	vec2 uScattering = vec2(rnd(seed), rnd(seed));
	f = BSDF_Sample_f(si.bsdf, si.wo, uScattering, wi, scatteringPdf);
	f *= abs(dot(wi, si.n));

	if (f != Spectrum(0.f) && scatteringPdf > 0) 
	{
		// Account for light contributions along new sampled direction _wi_
		float weight = 1;
		if (!sampledSpecular) 
		{
			lightPdf = spherePdf(vec3(light.position), light.radius, si.p, wi);
			if (lightPdf == 0) 
			{
				return Ld;
			}
			weight = powerHeuristic(1, scatteringPdf, 1, lightPdf);
		}

		// Find intersection and compute transmittance
		// cast shadow ray 
		float tMin   = 0.001;
		float tMax   = distance(samplePoint, si.p) + light.radius * 2;
		//vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
		vec3  origin =  si.p + wi * tMin;
		vec3  rayDir = wi;

#define CHECK_LIGHT_OCCLUSION 0
#if CHECK_LIGHT_OCCLUSION
		//TODO verify if the check for occlusion is really needed
		uint  flags  = gl_RayFlagsSkipClosestHitShaderEXT;
		isShadowed   = true;
		traceRayEXT(topLevelAS,  // acceleration structure
					flags,       // rayFlags
					0xFF,        // cullMask
					0,           // sbtRecordOffset
					0,           // sbtRecordStride
					1,           // missIndex
					origin,      // ray origin
					tMin,        // ray min range
					rayDir,      // ray direction
					tMax,        // ray max range
					1            // payload (location = 1)
		);
#else
		isShadowed   = false;
#endif
		
		bool foundSurfaceInteraction = intersectSphere(vec3(light.position), light.radius, origin, rayDir) != -1 && !isShadowed;
		// Add light contribution from material sampling
		Spectrum Li = Spectrum(0.f);
	 
		if (foundSurfaceInteraction) 
		{
			 //if an intersection between the ray spawned in the new direction and the light is found add its contribution
	
			Li = Spectrum(light.intensity);
		} 
	
		if (Li != Spectrum(0.f))
		{
			Ld += f * Li * weight / scatteringPdf;
		}
	}

return Ld;
}

Spectrum uniformSampleOneLight(in SurfaceInteraction si, inout uint seed)
{
	Light l = uni.lights[0];
	return estimateDirect(si, l, seed);
}

void main()
{
  // Object data
	ObjDesc    objResource = objDesc.i[gl_InstanceCustomIndexEXT];
	MatIndices matIndices  = MatIndices(objResource.materialIndexAddress);
	Materials  materials   = Materials(objResource.materialAddress);
	Indices    indices     = Indices(objResource.indexAddress);
	Vertices   vertices    = Vertices(objResource.vertexAddress);

	// Indices of the triangle
	ivec3 ind = indices.i[gl_PrimitiveID];

	// Vertex of the triangle
	Vertex v0 = vertices.v[ind.x];
	Vertex v1 = vertices.v[ind.y];
	Vertex v2 = vertices.v[ind.z];

	const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

	// Computing the normal at hit position
	vec3 normal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
	// Transforming the normal to world space
	normal = normalize(vec3(normal * gl_WorldToObjectEXT));

	// Computing the coordinates of the hit position
	vec3 worldPos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
	// Transforming the position to world space
	worldPos = vec3(gl_ObjectToWorldEXT * vec4(worldPos, 1.0));

	cLight.inHitPosition = worldPos;

	uint seed = prd.seed;  // We don't want to modify the PRD
	// Possibly add emitted light at intersection
	// in case bounce == 0 or previous bounce was specular

	// Material of the object
	int               matIdx = matIndices.i[gl_PrimitiveID];
	WaveFrontMaterial mat    = materials.m[matIdx];

	// COMPUTE BXDF
	BSDF bsdf;
	vec3 nx,ny;
	coordinateSystem(normal, nx, ny);
	ConstructBSDF(mat, pcRay, nx, ny, normal, bsdf);

	SurfaceInteraction si = SurfaceInteraction(worldPos, -gl_WorldRayDirectionEXT, normal, bsdf);

	// SAMPLE ILLUMINATION FROM ONE LIGHT TO FIND PATH CONTRIBUTION.
	// SKIP FOR SPECULAR MATERIAL
	if(!isSpecular(bsdf.bxdfs[0].reflectionType))
	{
		prd.hitValue = uniformSampleOneLight(si,seed);
	}

	// SAMPLE BSDF TO GET NEW PATH DIRECTION
	vec3 wo = normalize(-gl_WorldRayDirectionEXT);
	vec3 wi;
	float pdf;
	vec2 u = vec2(rnd(seed), rnd(seed));
	Spectrum f = BSDF_Sample_f(bsdf, wo, u, wi,pdf);

	if(f == Spectrum(0.f) || pdf == 0.f)
	{
		prd.done      = 1;
		//DEBUG CODE
		if(pcRay.debug == DEBUG_RCHIT + 1)
		{
			if(f == Spectrum(0.f))
			{
				prd.hitValue = vec3(1.f,0.f,0.f);
				prd.attenuation = Spectrum(1.f);
			}
			else if(pdf == 0.f)
			{
				prd.hitValue = vec3(0.f,1.f,0.f);
				prd.attenuation = Spectrum(1.f);
			}
			else
			{
				prd.hitValue = vec3(0.f,0.f,1.f);
				prd.attenuation = Spectrum(1.f);
			}
		}
		return;
	}
  
	prd.attenuation *= f * abs(dot(wi, si.n)) / pdf;

	prd.done      = 0;
	prd.rayOrigin = worldPos;
	prd.rayDir    = wi;

	//DEBUG CODE
	if(pcRay.debug == DEBUG_RCHIT)
	{
		if( uni.lights[0].type == 1.f )
		{
			prd.hitValue = vec3(0.f,0.f,1.f);
		}
		else if( uni.lights[0].type == 0.f )
		{
			prd.hitValue = vec3(0.f,1.f,0.f);
		}

		prd.done = 1;
		prd.attenuation = Spectrum(1.f);
		
		if( length(prd.attenuation)> 3.f)
		{
			//prd.hitValue = vec3(0.f,0.f,1.f);
		}
	}
}