#extension GL_EXT_debug_printf : enable

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
const uint BXDF_MICROFACET_REFLECTION = 2;

//#define MAX_ADD_FLOAT_DATA 5
//#define MAX_ADD_VEC3_DATA 3
//#define MAX_ADD_INT_DATA 2
#define MAX_BXDFS 1

#define TROWBRIDGEREITZ_DISTR 1

#define FR_DIELECTRIC 1
#define FR_CONDUCTOR 2

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//GENERAL UTILITY FUNCTIONS

void swap(inout float x1, inout float x2)
{
	float tmp = x1;
	x1 = x2;
	x2 = tmp;
}

//------------------------------------------------------------



//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//REFLECTION STRUCTS

struct Fresnel
{
	float etaIDielectric;
	float etaTDielectric;
	Spectrum etaIConductor;
	Spectrum etaTConductor;
	Spectrum kConductor;
	int type;
};

struct MicrofacetDistribution
{
	float alphax;
	float alphay;
	int type;
};


struct BxDF
{
	uint reflectionType;
	uint type;
	Spectrum R;
	Fresnel fr;
	MicrofacetDistribution distr;
//	float additionalFloatData[MAX_ADD_FLOAT_DATA];
//	vec3 additionalVec3Data[MAX_ADD_VEC3_DATA];
//	int additionalIntData[MAX_ADD_INT_DATA];
};

struct BSDF
{
  //WaveFrontMaterial mat;
  mat3 localToWorld;
  mat3 worldToLocal;
  BxDF bxdfs[MAX_BXDFS];
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
//------------------------------------------------------------



//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//BSDF UTILITY
bool isSpecular(const in uint reflectionType)
{
	return (reflectionType & BSDF_SPECULAR) != 0;
}

float powerHeuristic(const in int nf, const in float fPdf,const in int ng, const in float gPdf) 
{
    float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

// BSDF Inline Functions
float CosTheta(const in vec3 w) { return w.z; }
float Cos2Theta(const in vec3 w) { return w.z * w.z; }
float AbsCosTheta(const in vec3 w) { return abs(w.z); }
float Sin2Theta(const in vec3 w) {
    return max(0.f, 1.f - Cos2Theta(w));
}

float SinTheta(const in vec3 w) { return sqrt(Sin2Theta(w)); }

float TanTheta(const in vec3 w) { return SinTheta(w) / CosTheta(w); }

float Tan2Theta(const in vec3 w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

bool SameHemisphere(const in vec3 v1, const in vec3 v2) {
	return v1.z * v2.z > 0;
}

float CosPhi(const in vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : clamp(w.x / sinTheta, -1, 1);
}

float SinPhi(const in vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : clamp(w.y / sinTheta, -1, 1);
}

float Cos2Phi(const in vec3 w) { return CosPhi(w) * CosPhi(w); }

float Sin2Phi(const in vec3 w) { return SinPhi(w) * SinPhi(w); }

float CosDPhi(const in vec3 wa, const in vec3 wb) {
    float waxy = wa.x * wa.x + wa.y * wa.y;
    float wbxy = wb.x * wb.x + wb.y * wb.y;
    if (waxy == 0 || wbxy == 0)
        return 1;
    return clamp((wa.x * wb.x + wa.y * wb.y) / sqrt(waxy * wbxy), -1, 1);
}

vec3 Reflect(const in vec3 wo, const in vec3 n) {
    return -wo + 2.f * dot(wo, n) * n;
}

vec2 ConcentricSampleDisk(const in vec2 u) {
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

vec3 CosineSampleHemisphere(const in vec2 u) {
    vec2 d = ConcentricSampleDisk(u);
    float z = sqrt(max(0.f, 1 - d.x * d.x - d.y * d.y));
    return vec3(d.x, d.y, z);
	//return vec3(d.x, z, -d.y);
}

vec3 Faceforward(const in vec3 v1, const in vec3 v2)
{
	return (dot(v1, v2) < 0.f) ? -v1 : v1;
}
 


//------------------------------------------------------------



//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//FRESNEL

//Fresnel ConstructFresnel(const in BxDF bxdf)
//{
//	Fresnel fr;
//	fr.type = bxdf.additionalIntData[1];
//	switch(bxdf.additionalIntData[1])
//	{
//		case FR_DIELECTRIC:
//		{
//			fr.etaIDielectric = bxdf.additionalFloatData[1];
//			fr.etaTDielectric = bxdf.additionalFloatData[2];
//			break;
//		}
//		case FR_CONDUCTOR:
//		{
//			fr.etaIConductor =	bxdf.additionalVec3Data[0];
//			fr.etaTConductor =	bxdf.additionalVec3Data[1];
//			fr.kConductor =		bxdf.additionalVec3Data[2];
//			break;
//		}
//	}
//	return fr;
//}
//

Spectrum EvaluateFresnelDielectric(in float cosThetaI, in float etaI,in float etaT)
{								   
     cosThetaI = clamp(cosThetaI, -1.f, 1.f);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;

    if (!entering) {
        swap(etaI, etaT);
        cosThetaI = abs(cosThetaI);
    }

    // Compute _cosThetaT_ using Snell's law
    float sinThetaI = sqrt(max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1.f) return Spectrum(1.f);
    float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return Spectrum((Rparl * Rparl + Rperp * Rperp) / 2.f);
}

Spectrum EvaluateFresnelConductor(in float cosThetaI, in Spectrum etai,in Spectrum etat,in Spectrum k)
{	
    cosThetaI = clamp(cosThetaI, -1, 1);
    Spectrum eta = etat / etai;
    Spectrum etak = k / etai;

    float cosThetaI2 = cosThetaI * cosThetaI;
    float sinThetaI2 = 1. - cosThetaI2;
    Spectrum eta2 = eta * eta;
    Spectrum etak2 = etak * etak;

    Spectrum t0 = eta2 - etak2 - sinThetaI2;
    Spectrum a2plusb2 = sqrt(t0 * t0 + 4 * eta2 * etak2);
    Spectrum t1 = a2plusb2 + cosThetaI2;
    Spectrum a = sqrt(0.5f * (a2plusb2 + t0));
    Spectrum t2 = 2 * cosThetaI * a;
    Spectrum Rs = (t1 - t2) / (t1 + t2);

    Spectrum t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
    Spectrum t4 = t2 * sinThetaI2;
    Spectrum Rp = Rs * (t3 - t4) / (t3 + t4);

    return 0.5 * (Rp + Rs);
}



Spectrum EvaluateFresnel(const in Fresnel fr, in float cosThetaI)
{	
	switch(fr.type)
	{
		case  FR_DIELECTRIC:
		{
			return EvaluateFresnelDielectric(cosThetaI, fr.etaIDielectric, fr.etaTDielectric);
		}
		case FR_CONDUCTOR:
		{
			return EvaluateFresnelConductor(abs(cosThetaI), fr.etaIConductor, fr.etaTConductor, fr.kConductor);
		}
	}	
	return Spectrum(0.f);
}
//------------------------------------------------------------





//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//MICROFACET DISTRIBUTION

//MicrofacetDistribution ConstructMicrofacetDistribution(const in BxDF bxdf)
//{
//	MicrofacetDistribution dist;
//	float alpha = bxdf.additionalFloatData[0];
//	dist.alphax = alpha;
//	dist.alphay = alpha;
//	dist.type = bxdf.additionalIntData[0];
//	return dist;
//}

// TrowbridgeReitzDistribution
float TrowbridgeReitzDistribution_RoughnessToAlpha(in float roughness)
{
	roughness = max(roughness, 1e-3);
    float x = log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
           0.000640711f * x * x * x * x;
}

void TrowbridgeReitzSample11(const in float cosTheta, in float U1, in float U2, out float slope_x, out float slope_y) 
{
  // special case (normal incidence)
    if (cosTheta > .9999f) {
        float r = sqrt(U1 / (1 - U1));
        float phi = 6.28318530718f * U2;
        slope_x = r * cos(phi);
        slope_y = r * sin(phi);
        return;
    }

    float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
    float tanTheta = sinTheta / cosTheta;
    float a = 1 / tanTheta;
    float G1 = 2 / (1 + sqrt(1.f + 1.f / (a * a)));

    // sample slope_x
    float A = 2 * U1 / G1 - 1;
    float tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10) tmp = 1e10;
    float B = tanTheta;
    float D = sqrt(max(B * B * tmp * tmp - (A * A - B * B) * tmp, 0.f));
    float slope_x_1 = B * tmp - D;
    float slope_x_2 = B * tmp + D;
    slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

    // sample slope_y
    float S;
    if (U2 > 0.5f) {
        S = 1.f;
        U2 = 2.f * (U2 - .5f);
    } else {
        S = -1.f;
        U2 = 2.f * (.5f - U2);
    }
    float z =
        (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
        (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    slope_y = S * z * sqrt(1.f + slope_x * slope_x);
}

vec3 TrowbridgeReitzSample(const in vec3 wi, const in float alpha_x,  const in float alpha_y,  const in float U1, const in  float U2) 
{
    // 1. stretch wi
    vec3 wiStretched =
        normalize(vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    float slope_x, slope_y;
    TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2, slope_x, slope_y);

    // 3. rotate
    float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    // 5. compute normal
    return normalize(vec3(-slope_x, -slope_y, 1.));
}

vec3 TrowbridgeReitzDistribution_Sample_wh(const in MicrofacetDistribution distr, const in vec3 wo, const in vec2 u)
{
    bool flip = wo.z < 0;
    vec3 wh = TrowbridgeReitzSample(flip ? -wo : wo, distr.alphax, distr.alphay, u[0], u[1]);
    if (flip) wh = -wh;
	return wh;
}

float TrowbridgeReitzDistribution_D(const in MicrofacetDistribution distr, const in vec3 wh)
{
	float tan2Theta = Tan2Theta(wh);
	if(isinf(tan2Theta)) return 0.f;
	float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
	float e = (Cos2Phi(wh) / (distr.alphax * distr.alphax) + Sin2Phi(wh) / (distr.alphay * distr.alphay)) * tan2Theta;
	return 1.f / (PI * distr.alphax * distr.alphay * cos4Theta * (1.f + e) * (1.f + e));
}

float TrowbridgeReitzDistribution_Lambda(const in MicrofacetDistribution distr, const in vec3 w)
{
	float absTanTheta = abs(TanTheta(w));
	if(isinf(absTanTheta)) return 0.f;
	float alpha = sqrt(Cos2Phi(w) * distr.alphax * distr.alphax + Sin2Phi(w) * distr.alphay * distr.alphay);
	float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
	return (-1.f + sqrt(1.f + alpha2Tan2Theta)) / 2.f;
}

//Common interface
float MicrofacetDistribution_Lambda(const in MicrofacetDistribution distr, const in vec3 w)
{

	float Lambda;
	switch(distr.type)
	{
		case TROWBRIDGEREITZ_DISTR:
		{
			Lambda =  TrowbridgeReitzDistribution_Lambda(distr, w);
			break;
		}
	}

	return Lambda;
}

float MicrofacetDistribution_G1(const in MicrofacetDistribution distr, const in vec3 w)
{
    return 1.f / (1.f + MicrofacetDistribution_Lambda(distr, w));
}

float MicrofacetDistribution_G(const in MicrofacetDistribution distr, const vec3 wo, const vec3 wi)
{
	return 1.f / (1.f + MicrofacetDistribution_Lambda(distr, wo) + MicrofacetDistribution_Lambda(distr, wi));
}

float MicrofacetDistribution_D(const in MicrofacetDistribution distr, const in vec3 wh)
{
	float D;
	switch(distr.type)
	{
		case TROWBRIDGEREITZ_DISTR:
		{
			D =  TrowbridgeReitzDistribution_D(distr, wh);
			break;
		}
	}

	return D;
}

float MicrofacetDistribution_Pdf(const in MicrofacetDistribution distr, const in vec3 wo, const in vec3 wh)
								  {
   // if (sampleVisibleArea)
        return MicrofacetDistribution_D(distr, wh) * MicrofacetDistribution_G1(distr, wo) * abs(dot(wo, wh)) / AbsCosTheta(wo);
//    else
//        return D(wh) * AbsCosTheta(wh);
}

// TrowbridgeReitzDistribution
float MicrofacetDistribution_RoughnessToAlpha(const in MicrofacetDistribution distr, float roughness)
{
	float alpha;
	switch(distr.type)
	{
		case TROWBRIDGEREITZ_DISTR:
		{
			alpha =  TrowbridgeReitzDistribution_RoughnessToAlpha(roughness);
			break;
		}
	}

	return alpha;
}

vec3 MicrofacetDistribution_Sample_wh(const in MicrofacetDistribution distr, const in vec3 wo, const in vec2 u)
{
	vec3 wh;
	switch(distr.type)
	{
		case TROWBRIDGEREITZ_DISTR:
		{
			wh =  TrowbridgeReitzDistribution_Sample_wh(distr, wo, u);
			break;
		}
	}

	return wh;
}
//------------------------------------------------------------







//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//MAIN BSDF FUNCTIONS
void ConstructBSDF(const in WaveFrontMaterial mat, const in PushConstantRay pcRay, const in vec3 nx, const in vec3 ny, const in vec3 nz, out BSDF bsdf)
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
		case BXDF_MICROFACET_REFLECTION:
		{
			bsdf.bxdfsNum = 1;
			bsdf.bxdfs[0].type = BXDF_MICROFACET_REFLECTION;
			bsdf.bxdfs[0].reflectionType = BSDF_REFLECTION | BSDF_GLOSSY;
			bsdf.bxdfs[0].R = Spectrum(0.f,0.f,8.f);

			//Additional data are:
			//TrowbridgeReitzDistribution:
			bsdf.bxdfs[0].distr.type = TROWBRIDGEREITZ_DISTR;


			bsdf.bxdfs[0].fr.type = pcRay.fresnelType;
			switch(pcRay.fresnelType)
			{
				case FR_DIELECTRIC:
				{
					float alpha = TrowbridgeReitzDistribution_RoughnessToAlpha(pcRay.rough);
					bsdf.bxdfs[0].distr.alphax = alpha;
					bsdf.bxdfs[0].distr.alphay = alpha;

					bsdf.bxdfs[0].fr.etaIDielectric = 1.f;
					bsdf.bxdfs[0].fr.etaTDielectric = pcRay.etaTDielectric;
					break;
				}
				case FR_CONDUCTOR:
				{
					bsdf.bxdfs[0].distr.alphax = pcRay.rough;
					bsdf.bxdfs[0].distr.alphay = pcRay.rough;

					bsdf.bxdfs[0].fr.etaIConductor = vec3(1.f);
					bsdf.bxdfs[0].fr.etaTConductor = vec3(4.369683f, 2.916703f, 1.654701f);
					bsdf.bxdfs[0].fr.kConductor = vec3(5.206434f, 4.231365f, 3.754947);
					break;
				}
			}

			break;
		}
	};


}

//BSDF FUNCTIONS
Spectrum BxDF_f(const in BxDF bxdf, const in vec3 wo, const in vec3 wi)
{
	Spectrum f;

	switch(bxdf.type)
	{
		case BXDF_LAMBERTIAN_REFLECTION:
		{
			f = bxdf.R * INV_PI;
			break;
		}
		case BXDF_MICROFACET_REFLECTION:
		{
			
			float cosThetaO = AbsCosTheta(wo); 
			float cosThetaI = AbsCosTheta(wi);
			vec3 wh = wi + wo;
			//Handle degenerate case for microfacet reflection 
			if(cosThetaI == 0.f || cosThetaO == 0.f)
			{
				return Spectrum(0.f);
			}
			if(wh == vec3(0.f))
			{
				return Spectrum(0.f);
			}

			wh = normalize(wh);
			
			//Spectrum F = EvaluateFresnel(bxdf.fr, dot(wi, Faceforward(wh, vec3(0,0,1))));
			Spectrum F = EvaluateFresnel(bxdf.fr, dot(wi, wh));
			
			float D = MicrofacetDistribution_D(bxdf.distr, wh);
			float G = MicrofacetDistribution_G(bxdf.distr, wo, wi);
			f =   bxdf.R * D * G * F / (4.f * cosThetaI * cosThetaO);
			break;
		}
	}
	return f;
}

Spectrum BSDF_f(const in BSDF bsdf, const in vec3 woW, const in vec3 wiW)
{
	Spectrum f;
	//transform woW, wiW to local Coordinates
	vec3 wo = bsdf.worldToLocal * woW;
	vec3 wi = bsdf.worldToLocal * wiW;

	return BxDF_f(bsdf.bxdfs[0], wo, wi);
}

float BxDF_Pdf(const in BxDF bxdf, const in vec3 wo, const in vec3 wi)
{
	float pdf;

	switch(bxdf.type)
	{
		case BXDF_LAMBERTIAN_REFLECTION:
		{
			pdf = SameHemisphere(wo, wi) ? AbsCosTheta(wi) * INV_PI : 0;
			break;
		}
		case BXDF_MICROFACET_REFLECTION:
		{
			//MicrofacetDistribution distr = ConstructMicrofacetDistribution(bxdf);
			if(!SameHemisphere(wo, wi))
			{
				 return 0.f;
			}

			//Compute pdf of wi for microfacet reflection 
			vec3 wh = normalize(wo + wi);
			pdf = MicrofacetDistribution_Pdf(bxdf.distr, wo, wh) / (4.f * dot(wo, wh));
			break;
		}
	}
	return pdf;
}


float BSDF_Pdf(const in BSDF bsdf, const in vec3 woW, const in vec3 wiW)
{
	float pdf;
	//transform woW, wiW to local Coordinates
	vec3 wo = bsdf.worldToLocal * woW;
	vec3 wi = bsdf.worldToLocal * wiW;

	return BxDF_Pdf(bsdf.bxdfs[0],wo, wi);
}

Spectrum BxDF_Sample_f(const in BxDF bxdf, const in vec3 wo, const in vec2 u, out vec3 wiOut, out float pdf)
{
	Spectrum f;
	//Not sure if an out vector can be passed to another function as in vec
	//so use wi to be sure
	vec3 wi;
	switch(bxdf.type)
	{
		case BXDF_LAMBERTIAN_REFLECTION:
		{
			// Cosine-sample the hemisphere, flipping the direction if necessary
			// to make sure wi and wo are on the same hemisphere
			wi = CosineSampleHemisphere(u);
			//if (!sameHemisphere(wo, wi))
			if (wo.z < 0)
			{
				wi.z *= -1;
			}

			pdf = BxDF_Pdf(bxdf, wo, wi);
			f =	  BxDF_f(bxdf, wo, wi);
			break;
		}
		case BXDF_MICROFACET_REFLECTION:
		{
			//MicrofacetDistribution distr = ConstructMicrofacetDistribution(bxdf);

			//Sample microfacet orientation wh and reflected direction wi
			vec3 wh = MicrofacetDistribution_Sample_wh(bxdf.distr, wo, u);
			if (dot(wo, wh) < 0) return Spectrum(0.f);   // Should be rare

			wi = Reflect(wo, wh);
			if(!SameHemisphere(wo, wi))	return Spectrum(0.f);	//ok if it happens


			//Compute pdf of wi for microfacet reflection 
			pdf = MicrofacetDistribution_Pdf(bxdf.distr, wo, wh) / (4.f * dot(wo, wh));
			f =	  BxDF_f(bxdf, normalize(wo),normalize(wi));

			break;
		}
	}
	wiOut = wi;
	return f;
}

Spectrum BSDF_Sample_f(const in BSDF bsdf, const in vec3 woW, const in vec2 u, out vec3 wiW, out float pdfOut)
{
	Spectrum f;
	//transform woW, wiW to local Coordinates
	vec3 wo = bsdf.worldToLocal * woW;
	vec3 wi;
	float pdf;

	f = BxDF_Sample_f(bsdf.bxdfs[0], wo, u, wi, pdf);

	pdfOut = pdf;
	wiW = bsdf.localToWorld * wi;
	return f;
}
