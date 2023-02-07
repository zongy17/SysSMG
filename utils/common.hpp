#ifndef SOLID_COMMON_HPP
#define SOLID_COMMON_HPP

#ifndef KSP_BIT
#define KSP_BIT 64
#endif
#ifndef PC_DATA_BIT
#define PC_DATA_BIT 64
#endif
#ifndef PC_CALC_BIT
#define PC_CALC_BIT 64
#endif

#if KSP_BIT==80
#define KSP_TYPE long double
#define KSP_MPI_TYPE MPI_LONG_DOUBLE
#elif KSP_BIT==64
#define KSP_TYPE double
#define KSP_MPI_TYPE MPI_DOUBLE
#elif KSP_BIT==32
#define KSP_TYPE float
#define KSP_MPI_TYPE MPI_FLOAT
#endif

#if PC_DATA_BIT==80
#define PC_DATA_TYPE long double 
#elif PC_DATA_BIT==64
#define PC_DATA_TYPE double 
#elif PC_DATA_BIT==32
#define PC_DATA_TYPE float
#elif PC_DATA_BIT==16
#define PC_DATA_TYPE __fp16
#endif

#if PC_CALC_BIT==80
#define PC_CALC_TYPE long double 
#elif PC_CALC_BIT==64
#define PC_CALC_TYPE double 
#elif PC_CALC_BIT==32
#define PC_CALC_TYPE float
#elif PC_CALC_BIT==16
#define PC_CALC_TYPE __fp16
#endif

#define IDX_TYPE int

#ifndef NUM_DOF
#define NUM_DOF 3
#endif

// #define NDEBUG

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include <string.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

enum NEIGHBOR_ID {I_L, I_U, J_L, J_U, K_L, K_U, NUM_NEIGHBORS};
typedef enum {VERT} LINE_DIRECTION;
typedef enum {XZ} PLANE_DIRECTION;

#define MIN(a, b) ((a) > (b)) ? (b) : (a)

template<typename T>
bool check_power2(T num) {
    while (num > 1) {
        T new_num = num >> 1;
        if (num != new_num * 2) return false;
        num = new_num;
    }
    return true;
}

const IDX_TYPE stencil_offset_3d7[7 * 3] = {
    // y , x , z
    -1,  0,  0, // 0
    0, -1,  0, // 1
    0,  0, -1, // 2
    0,  0,  0, // 3
    0,  0,  1, // 4
    0,  1,  0, // 5
    1,  0,  0  // 6
};

const IDX_TYPE stencil_offset_3d15[15 * 3] = {
    // j  i  k
    -1, -1, -1,// 0
    -1, -1,  0,// 1
    -1,  0, -1,// 2
    -1,  0,  0,// 3
     0, -1, -1,// 4
     0, -1,  0,// 5
     0,  0, -1,// 6
     0,  0,  0,// 7: center
     0,  0,  1,
     0,  1,  0,
     0,  1,  1,
     1,  0,  0,// 11
     1,  0,  1,// 12
     1,  1,  0,
     1,  1,  1
};

const IDX_TYPE stencil_offset_3d27[27 * 3] = {
    // y , x , z
    -1, -1, -1,
    -1, -1,  0,
    -1, -1,  1,
    -1,  0, -1,
    -1,  0,  0,
    -1,  0,  1,
    -1,  1, -1,
    -1,  1,  0,
    -1,  1,  1,

    0, -1, -1,
    0, -1,  0,
    0, -1,  1,
    0,  0, -1,
    0,  0,  0,
    0,  0,  1,
    0,  1, -1,
    0,  1,  0,
    0,  1,  1,

    1, -1, -1,
    1, -1,  0,
    1, -1,  1,
    1,  0, -1,
    1,  0,  0,
    1,  0,  1,
    1,  1, -1,
    1,  1,  0,
    1,  1,  1
};


#define CHECK_HALO(x , y) \
    assert((x).halo_x == (y).halo_x  &&  (x).halo_y == (y).halo_y  &&  (x).halo_z == (y).halo_z);

#define CHECK_LOCAL_HALO(x , y) \
    assert((x).local_x == (y).local_x && (x).local_y == (y).local_y && (x).local_z == (y).local_z && \
           (x).halo_x == (y).halo_x  &&  (x).halo_y == (y).halo_y  &&  (x).halo_z == (y).halo_z);

#define CHECK_INOUT_DIM(x , y) \
    assert( (x).input_dim[0] == (y).input_dim[0] && (x).input_dim[1] == (y).input_dim[1] && \
            (x).input_dim[2] == (y).input_dim[2] && \
            (x).output_dim[0] == (y).output_dim[0] && (x).output_dim[1] == (y).output_dim[1] && \
            (x).output_dim[2] == (y).output_dim[2] );

#define CHECK_OFFSET(x , y) \
    assert((x).offset_x == (y).offset_x && (x).offset_y == (y).offset_y && (x).offset_z == (y).offset_z);

#define CHECK_INPUT_DIM(x , y) \
    assert( (x).input_dim[0] == (y).global_size_x && (x).input_dim[1] == (y).global_size_y && \
            (x).input_dim[2] == (y).global_size_z );

#define CHECK_OUTPUT_DIM(x , y) \
    assert( (x).output_dim[0] == (y).global_size_x && (x).output_dim[1] == (y).global_size_y && \
            (x).output_dim[2] == (y).global_size_z );

#define CHECK_VEC_GLOBAL(x, y) \
    assert( (x).global_size_x == (y).global_size_x && (x).global_size_y == (y).global_size_y && \
            (x).global_size_z == (y).global_size_z );

double wall_time() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1. * t.tv_sec + 1.e-9 * t.tv_nsec;
}

#if defined(__aarch64__)
#define barrier() __asm__ __volatile__("dmb" ::: "memory")
#define smp_mb()  __asm__ __volatile__("dmb ish" ::: "memory")
#define smp_wmb() __asm__ __volatile__("dmb ishst" ::: "memory")
#define smp_rmb() __asm__ __volatile__("dmb ishld" ::: "memory")
#else
// #error No architecture detected.
#endif

#include <unistd.h>
#include <sched.h>
#include <pthread.h>

void print_affinity() {
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);// 这个节点上on_line的有多少个cpu核
    printf("affinity_cpu=%02d of %ld ", sched_getcpu(), nproc);// sched_getcpu()返回这个线程绑定的核心id
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity error");
        exit(1);
    }
    printf("affinity_mask=");
    for (int i = 0; i < nproc; i++) printf("%d", CPU_ISSET(i, &mask));
    printf("\n");
}

template<typename idx_t>
void uniformly_distributed_integer(idx_t M, idx_t N, idx_t * arr)
{// assert arr's space allocated of N
    assert(M >= N);
    idx_t gap = M / N;
    idx_t rem = M - gap * N;

    arr[0] = 0;
    for (idx_t i = 0; i < N - 1; i++)
        arr[i + 1] = arr[i] + gap + ((i<rem) ? 1 : 0);    
}

template <int len, typename data_t>
struct PointerWrapper
{
    data_t val[len];
    PointerWrapper(const data_t init_v) {
        for (int i = 0; i < len; i++)
            val[i] = init_v;
    }
    PointerWrapper& operator+=(const PointerWrapper &b) {
       for (int i = 0; i < len; i++)
         val[i] += b.val[i];
       return *this;
    }
    void mla(const data_t wt, const data_t * p) {
      for (int i = 0; i < len; i++)
        val[i] += wt * p[i];
    }
    PointerWrapper operator*(const data_t coeff){
        PointerWrapper ret(*this);
        for (int i = 0; i < len; i++)
            ret.val[i] *= coeff;
        return ret;
    }
    void store(data_t * p) {
      for (int i = 0; i < len; i++)
        p[i] = val[i];
    }
    void print(){
        for (int i = 0; i < len; i++)
            printf("%.4f ", val[i]);
        printf("\n");
    }
};


#endif