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

#define WG 16
#define SIMD 16

extern void lib_func(int q, const unsigned int N, __global const float *m, __global const particle_t *p, __global coord3d_t *a);

__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((num_simd_work_items(SIMD)))
kernel void kernel_lib(const unsigned int N, __global const float * restrict m, __global const particle_t * restrict in_p, __global coord3d_t * restrict a){

    int gid = get_global_id(0);
    lib_func(gid, N, m, in_p, a);

}
