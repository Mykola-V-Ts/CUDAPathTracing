# Path tracing renderer

The main purpose of this project is to deepen my understanding of ray tracing by implementing algorithm from Peter Shirley's book ["Ray Tracing in One Weekend"](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

Axis aligned planes and emmissive material are made according to later chapters of [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html)
<br/>To leverage the performance benefits of CUDA I used Roger Allen's [blog post](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) on the same topic.

**Please note, that this program is not an original design but rather an educational exercise.**

## Features

* Basic unidirectional path tracing algorithm as described in [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
* Spheres and planes as geometric primitives
* Lambertian, metallic and emissive materials
* NVDIA GPU support only

## Acknowledgments

* Peter Shirley for his excellent [Ray Tracing in One Weekend Series](https://raytracing.github.io/)
* Roger Allen for accessible introduction to CUDA
