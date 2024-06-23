
#include <iostream>
#include <time.h>
#include <float.h>
#include <fstream>

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "plane.h"
#include "hittable.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"

__device__ vec3 ray_color(const ray& r, hittable** scene, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1, 1, 1);
    vec3 emition = vec3(0, 0, 0);

    for (int i = 0; i < 12; i++) {
        hit_record rec;

        if ((*scene)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;  

            // If the material didn't produce scattered ray then it is emmisive. Calculating color and exiting loop
            if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered, local_rand_state)) {
                emition = rec.mat_ptr->emitted();
                return emition * cur_attenuation;
            }

            // Adjusting attenuation based on current material
            cur_attenuation *= attenuation;
            cur_ray = scattered;
        }
        else {
            // Ray went out of bounds of the scene. Calculating color based on environment properties
            return vec3(0.0, 0.0, 0.0); // Current scene requires pitch black environment but I would put blue gradient here if wanted to simulate sky
        }
    }
    // Ray haven't found any light source and got lost somewhere in the scene geometry
    return vec3(0.0, 0.0, 0.0);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hittable** scene, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // Aborting the threads outside of the image bounds
    if ((i >= max_x) || (j >= max_y)) return;

    // Initializing rand_state for the thread
    int pixel_index = j * max_x + i;
    curand_init(0, pixel_index, 0, &rand_state[pixel_index]);
    curandState local_rand_state = rand_state[pixel_index];

    vec3 col(0, 0, 0);

    // Shooting rays from random points on the pixel, averaging the result
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += ray_color(r, scene, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;

    // Storing gamma corrected color to the frame buffer
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void create_scene(hittable** d_list, hittable** d_scene, camera** d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        
        // Materials
        auto material_grey = new lambertian(vec3(0.99, 0.99, 0.99));
        auto material_emissive = new diffuse_light(vec3(1, 1, 1));
        auto material_green = new lambertian(vec3(0.01, 0.8, 0.01));
        auto material_red = new lambertian(vec3(0.8, 0.01, 0.01));
        auto material_chrome = new metal(vec3(0.8, 0.8, 0.8), 0.0);

        int i = 0;
        
        // Light source
        d_list[i++] = new xz_plane(-7, 7, 1, 15, 7.999, material_emissive);

        // Walls of the room
        d_list[i++] = new xz_plane(-8, 8, 0, 16, 8, material_grey);
        d_list[i++] = new xz_plane(-8, 8, 0, 16, 0, material_grey);
        d_list[i++] = new yz_plane(0, 8, 0, 16, 8, material_green);
        d_list[i++] = new yz_plane(0, 8, 0, 16, -8, material_red);
        d_list[i++] = new xy_plane(-8, 8, 0, 8, 16, material_grey);

        // Cube
        d_list[i++] = new xz_plane(1, 5, 11, 15, 4, material_grey);
        d_list[i++] = new yz_plane(0, 4, 11, 15, 5, material_grey);
        d_list[i++] = new yz_plane(0, 4, 11, 15, 1, material_grey);
        d_list[i++] = new xy_plane(1, 5, 0, 4, 11, material_grey);
        d_list[i++] = new xy_plane(1, 5, 0, 4, 15, material_grey);

        // Spheres
        d_list[i++] = new sphere(vec3(-2, 1.5, 8), 1.5, material_grey);
        d_list[i++] = new sphere(vec3(1.5, 0.6, 5), 0.6, material_grey);
        d_list[i++] = new sphere(vec3(0, 0.6, 6), 0.3, material_chrome);
        
        *d_scene = new hittable_list(d_list, i);

        // Setting up the camera
        vec3 lookfrom(0, 1, 0);
        vec3 lookat(0, 1, 1);

        float dist_to_focus = (lookfrom - lookat).length(); // Not needed for current scene
        float aperture = 0.1;                               // Not needed for current scene
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            50,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

int main() {
    // Image size
    int nx = 1920;
    int ny = 1080;

    // Samples per pixel
    int ns = 1024;

    // Block size (5x5 is optimal for my GPU)
    int tx = 5;
    int ty = 5;

    std::cout << "Image size: " << nx << "x" << ny << std::endl;
    std::cout << "Samples: " << ns << std::endl;
    std::cout << "Blocks: " << tx << "x" << ty << std::endl;

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Allocating frame buffer
    vec3* fb;
    cudaMallocManaged((void**)&fb, fb_size);

    // Allocating random state
    curandState* d_rand_state;
    cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState));

    // Allocating space for scene props
    hittable** d_list;
    int num_hittables = 14;
    cudaMalloc((void**)&d_list, num_hittables * sizeof(hittable*));
    hittable** d_scene;
    cudaMalloc((void**)&d_scene, sizeof(hittable*));
    camera** d_camera;
    cudaMalloc((void**)&d_camera, sizeof(camera*));
    cudaDeviceSynchronize();

    // Fill out the scene
    create_scene<<<1, 1>>>(d_list, d_scene, d_camera, nx, ny);
    cudaDeviceSynchronize();

    clock_t start, stop;
    start = clock();

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    // Launch calculation
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_scene, d_rand_state);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time: " << timer_seconds << " seconds\n";

    // Outputting image to ppm file
    std::ofstream output("image.ppm");
    output << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            output << ir << " " << ig << " " << ib << "\n";
        }
    }
    output.close();

    cudaDeviceSynchronize();
    cudaDeviceReset();
}