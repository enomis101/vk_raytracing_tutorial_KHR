#include "host_device.h"

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

#define MAX_ADD_FLOAT_DATA 5
#define MAX_ADD_VEC3_DATA 1
#define MAX_BXDFS 1

struct BXDF
{
	uint reflectionType;
	uint type;
	Spectrum R;
	float additionalFloatData[MAX_ADD_FLOAT_DATA];
	vec3 additionalVec3Data[MAX_ADD_VEC3_DATA];
};

struct BSDF
{
  //WaveFrontMaterial mat;
  mat3 localToWorld;
  mat3 worldToLocal;
  BXDF bxdfs[MAX_BXDFS];
  int bxdfsNum;
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
	mat4x3 transform;
	float radius;
	float intensity;
};

bool isSpecular(in uint reflectionType)
{
	return (reflectionType & BSDF_SPECULAR) != 0;
}

float powerHeuristic(in int nf, in float fPdf,in int ng, in float gPdf) 
{
    float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

void constructBSDF(in WaveFrontMaterial mat, in vec3 nx, in vec3 ny, in vec3 nz, out BSDF bsdf)
{
	//Local to World matrix has axis as column
	mat3 localToWorld = mat3(nx,ny,nz);
	bsdf.localToWorld = localToWorld;
	bsdf.worldToLocal = transpose(localToWorld);

	switch(mat.bxdf)
	{
		case BXDF_LAMBERTIAN_REFLECTION:
		{
			bsdf.bxdfsNum = 1;
			bsdf.bxdfs[0].type = BXDF_LAMBERTIAN_REFLECTION;
			bsdf.bxdfs[0].reflectionType =BSDF_REFLECTION | BSDF_DIFFUSE;
			bsdf.bxdfs[0].R = mat.diffuse;
			break;
		}
	};


}

//UTILITY
float absCosTheta(in vec3 v)
{
	return abs(v.z);
}

bool sameHemisphere(in vec3 v1, in vec3 v2) {
	return v1.z * v2.z > 0;
}

vec2 concentricSampleDisk(in vec2 u) {
    // Map uniform random numbers to $[-1,1]^2$
    vec2 uOffset = 2.f * u - vec2(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return vec2(0, 0);

    // Apply concentric mapping to point
    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = PI_OVER4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PI_OVER2 - PI_OVER4 * (uOffset.x / uOffset.y);
    }
    return r * vec2(cos(theta), sin(theta));
}

vec3 cosineSampleHemisphere(in vec2 u) {
    vec2 d = concentricSampleDisk(u);
    float z = sqrt(max(0.f, 1 - d.x * d.x - d.y * d.y));
    return vec3(d.x, d.y, z);
	//return vec3(d.x, z, -d.y);
}

//BSDF FUNCTIONS

Spectrum getBSDFValueLocal(in BSDF bsdf, in vec3 wo, in vec3 wi)
{
	Spectrum f;

	switch(bsdf.bxdfs[0].type)
	{
	case BXDF_LAMBERTIAN_REFLECTION:
	{
		f = bsdf.bxdfs[0].R * INV_PI;
		break;
	}
	};
	return f;
}

Spectrum getBSDFValue(in BSDF bsdf, in vec3 woW, in vec3 wiW)
{
	Spectrum f;
	//transform woW, wiW to local Coordinates
	vec3 wo = bsdf.worldToLocal * woW;
	vec3 wi = bsdf.worldToLocal * wiW;

	return getBSDFValueLocal(bsdf, wo, wi);
}

float getBSDFPdfLocal(in BSDF bsdf, in vec3 wo, in vec3 wi)
{
	float pdf;

	switch(bsdf.bxdfs[0].type)
	{
	case BXDF_LAMBERTIAN_REFLECTION:
	{
		pdf = sameHemisphere(wo, wi) ? absCosTheta(wi) * INV_PI : 0;
		break;
	}
	};
	return pdf;
}


float getBSDFPdf(in BSDF bsdf, in vec3 woW, in vec3 wiW)
{
	float pdf;
	//transform woW, wiW to local Coordinates
	vec3 wo = bsdf.worldToLocal * woW;
	vec3 wi = bsdf.worldToLocal * wiW;

	return getBSDFPdfLocal(bsdf,wo, wi);
}


Spectrum sampleBSDF(in BSDF bsdf, in vec3 woW, in vec2 u, out vec3 wiW, out float pdf)
{
	Spectrum f;
	//transform woW, wiW to local Coordinates
	vec3 wo = bsdf.worldToLocal * woW;
	vec3 wi;
	switch(bsdf.bxdfs[0].type)
	{
	case BXDF_LAMBERTIAN_REFLECTION:
	{
		// Cosine-sample the hemisphere, flipping the direction if necessary
		// to make sure wi and wo are on the same hemisphere
		wi = cosineSampleHemisphere(u);
		//if (!sameHemisphere(wo, wi))
		if (wo.z < 0)
		{
			wi.z *= -1;
		}
		break;
	}
	};

	wiW = bsdf.localToWorld * wi;
	pdf = getBSDFPdfLocal(bsdf, wo, wi);
	f = getBSDFValueLocal(bsdf, wo, wi);
	return f;
}
