#ifndef PLANEH
#define PLANEH

#include "hittable.h"

class xy_plane : public hittable {
public:
    __device__ xy_plane() {}
    __device__ xy_plane(float _x0, float _x1, float _y0, float _y1, float _k, material* mat) : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat_ptr(mat) {};
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

    material* mat_ptr;
    float x0, x1, y0, y1, k;
};

class xz_plane : public hittable {
public:
    __device__ xz_plane() {}
    __device__ xz_plane(float _x0, float _x1, float _z0, float _z1, float _k, material* mat) : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {};
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

    material* mat_ptr;
    float x0, x1, z0, z1, k;
};

class yz_plane : public hittable {
public:
    __device__ yz_plane() {}
    __device__ yz_plane(float _y0, float _y1, float _z0, float _z1, float _k, material* mat) : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {};
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

    material* mat_ptr;
    float y0, y1, z0, z1, k;
};

__device__ bool xy_plane::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    float t = (k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max)
        return false;
    float x = r.origin().x() + t * r.direction().x();
    float y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    vec3 outward_normal = vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_parameter(t);
    return true;
}

__device__ bool xz_plane::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    float t = (k - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max)
        return false;
    float x = r.origin().x() + t * r.direction().x();
    float z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    vec3 outward_normal = vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_parameter(t);
    return true;
}

__device__ bool yz_plane::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    float t = (k - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max)
        return false;
    float y = r.origin().y() + t * r.direction().y();
    float z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    vec3 outward_normal = vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_parameter(t);
    return true;
}

#endif