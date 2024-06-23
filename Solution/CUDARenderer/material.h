#ifndef MATERIALH
#define MATERIALH

#include "ray.h"
#include "hittable.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

class material {
public:
    __device__ virtual vec3 emitted() const { return vec3(0, 0, 0); }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const = 0;
};

class lambertian : public material {
public:
    __device__ lambertian(const vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const override {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state); // Diffusing outcoming light
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }
    vec3 albedo;
};

class metal : public material {
public:
    __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal); // Calculating bounced ray
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state)); // Diffusing outcoming light with roughness multiplier
        attenuation = albedo;
        return true;
    }
    vec3 albedo;
    float fuzz;
};

class diffuse_light : public material {
public:
    __device__ diffuse_light(const vec3& e) : emit(e) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const override { return false; } // Stop further light bounces
    __device__ virtual vec3 emitted() const override { return emit; }

    vec3 emit;
};

#endif