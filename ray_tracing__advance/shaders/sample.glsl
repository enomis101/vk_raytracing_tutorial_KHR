#include "host_device.h"
//#include "raycommon.glsl"

void coordinateSystem(in vec3 v1, out vec3 v2, out vec3 v3)
{
  if(abs(v1.x) > abs(v1.y))
    v2 =vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
  else
    v2 = vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
  v3 = cross(v1, v2);
}

vec3 sphericalDirection(in float sinTheta, in float cosTheta, in float phi, in vec3 x,  in vec3 y, in vec3 z)
{
    return sinTheta * cos(phi) * x + sinTheta * sin(phi) * y + cosTheta * z;
}

float uniformConePdf(in float cosThetaMax) {
    return 1 / (2 * PI * (1 - cosThetaMax));
}

float spherePdf(in vec3 sphereCenter,in float sphereRadius, in vec3 p, in vec3 wi) 
{
    // Return uniform PDF if point is inside sphere
    if (abs(distance(p, sphereCenter)) <= sphereRadius)
        return 0.f;

    // Compute general sphere PDF
    float sinThetaMax2 = sphereRadius / abs(distance(p, sphereCenter));
    float cosThetaMax = sqrt(max(0.f, 1 - sinThetaMax2));
    return uniformConePdf(cosThetaMax);
}

// Ray-Sphere intersection
// http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/
float intersectSphere(in vec3 sphereCenter, in float sphereRadius, in vec3 rayOrigin, in vec3 rayDirection)
{
  vec3  oc           = rayOrigin - sphereCenter;
  float a            = dot(rayDirection, rayDirection);
  float b            = 2.0 * dot(oc, rayDirection);
  float c            = dot(oc, oc) - sphereRadius * sphereRadius;
  float discriminant = b * b - 4 * a * c;
  if(discriminant < 0)
  {
    return -1.0;
  }
  else
  {
    return (-b - sqrt(discriminant)) / (2.0 * a);
  }
}

/*
Compute a sample on a sphere given a point from which to sample
arguments:
sphere: center (in World Coordinates) and radius for the sphere to saple
sphereTransform: needed to transform back to World coordinate the sampled points
inPoint: point from which to sample
u: vec2 of 2 random number
out samplePoint: the sample points returned (in World Coordinates)

return:
true if success and false instead
*/
//bool sampleSphere(in Sphere sphere, in mat4 sphereTransform, in mat4 sphereInverseTransposeTransform, in vec3 inPoint, in vec2 u, out vec3 samplePoint, out vec3 sampleNormal)
bool sampleSphere(in vec3 sphereCenter, in float sphereRadius, in mat4x3 sphereTransform, in vec3 inPoint, in vec2 u, out vec3 samplePoint, out vec3 sampleNormal, out float pdf)
{
	//Compute coordinate system for sphere sampling
	// #Assume RHCS
	//(different from pbrt) which use LHCS

	vec3 pCenter = sphereCenter;
	//Use P - C instead of C - P (different from pbrt)
	vec3 pc = normalize(inPoint - pCenter);
	vec3 pcX, pcY;
	//Construct a coordinate system with pc as the axis
	coordinateSystem(pc, pcX,pcY);

	//For now return false if point is inside the sphere
	// #TODO sample uniformly on the sphere instead
	if(distance(inPoint, sphereCenter) <= sphereRadius)
	{
		return false;
	}
	
	//Sample sphere uniformly inside subtended cone

	//Compute theta and phi values for sample in cone
	float dc = distance(inPoint,sphereCenter);
	float sinThetaMax2 = sphereRadius * sphereRadius / (dc * dc);
	float cosThetaMax = sqrt(max(0.f, 1.f - sinThetaMax2));
	float cosTheta = (1.f - u[0]) + u[0] * cosThetaMax;
	float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
	float phi = u[1] * 2 * PI;

	//Compute angle alpha from the center of sphere to sampled point on surface
	float ds = dc * cosTheta - sqrt(max(0.f, sphereRadius * sphereRadius - dc * dc * sinTheta * sinTheta));
	float cosAlpha = (dc * dc + sphereRadius * sphereRadius - ds * ds) / (2 * dc * sphereRadius);
	float sinAlpha = sqrt(max(0.f, 1 - cosAlpha * cosAlpha));
	
	//Compute surface normal and sampled point on sphere
	vec3 nObj = sphericalDirection(sinAlpha, cosAlpha, phi, pcX, pcY, pc);
	vec3 pObj = sphereRadius * nObj;	//local coordinates

	//Transform normal and sampled point to world coordinates 
	//Because the orientation of a sphere is useless no need to transform it
	//Translate only
	samplePoint =  vec3(sphereTransform * vec4(pObj, 1.f));
	//sampleNormal = sphereInverseTransposeTransform * vec4(nObj, 0.f);

	// Uniform cone PDF.
    pdf = 1.f / (2.f * PI * (1 - cosThetaMax));
	return true;
}