// define library function (HLS)
#include "HLS/math.h"

#define OCL_ADDRSP_CONSTANT const __attribute__((address_space(2)))
#define OCL_ADDRSP_GLOBAL __attribute__((address_space(1)))
#define OCL_ADDRSP_PRIVATE __attribute__((address_space(0)))
#define OCL_ADDRSP_LOCAL __attribute__((address_space(3)))


typedef struct {
    float x;
    float y;
    float z;
    float padding;
} coord3d_t;

typedef struct {
    coord3d_t p;
    coord3d_t v;
} particle_t;

#define EPS 100

extern "C" void lib_func(int q, const unsigned int N, OCL_ADDRSP_GLOBAL const float *m, OCL_ADDRSP_GLOBAL const particle_t *p, OCL_ADDRSP_GLOBAL coord3d_t *a) { 

	float pqpx = p[q].p.x;
	float pqpy = p[q].p.y;
	float pqpz = p[q].p.z;
	float aqx = 0.0f; float aqy = 0.0f; float aqz = 0.0f;
        for (unsigned int j = 0; j < N; j++) {
            float rx = p[j].p.x - pqpx;
            float ry = p[j].p.y - pqpy;
            float rz = p[j].p.z - pqpz;
            float dd = rx*rx + ry*ry + rz*rz + EPS;
            float d = 1.0f/ (dd*sqrtf(dd));
            float s = m[j] * d;
            aqx += rx * s;
            aqy += ry * s;
            aqz += rz * s;
        }
	a[q].x = aqx;
	a[q].y = aqy;
	a[q].z = aqz;
}
