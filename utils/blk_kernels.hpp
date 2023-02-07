#ifndef SYSSMG_BLK_KERNELS_HPP
#define SYSSMG_BLK_KERNELS_HPP

#include "string.h"

template<typename idx_t, typename data_t, int dof>
inline void vec_zero(data_t * vec) {
    #pragma GCC unroll (4)
    for (idx_t f = 0; f < dof; f++)
        vec[f] = 0.0;
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
inline void matvec_mla(const data_t * mat, const calc_t * in, calc_t * out) {
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++) {
        calc_t tmp = 0.0;
        const data_t * vals = mat + r * dof;
        #pragma GCC unroll (4)
        for (idx_t c = 0; c < dof; c++)
            tmp += vals[c] * in[c];
        out[r] += tmp;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
inline void matvec_mla(const data_t * mat, const data_t * vec, const calc_t * in, calc_t * out) {
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++) {
        calc_t tmp = 0.0;
        const data_t * vals = mat + r * dof;
        #pragma GCC unroll (4)
        for (idx_t c = 0; c < dof; c++)
            tmp += vals[c] * in[c] * vec[c];
        out[r] += tmp;
    }
}

// out[] = wt * mat[][] * in[]
template<typename idx_t, typename data_t, typename calc_t, int dof>
inline void matvec_mul(const data_t * mat, const calc_t * in, calc_t * out, calc_t wt) {
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++) {
        calc_t tmp = 0.0;
        const data_t * vals = mat + r * dof;
        #pragma GCC unroll (4)
        for (idx_t c = 0; c < dof; c++)
            tmp += vals[c] * in[c];
        out[r] = wt * tmp;
    }
}

// out[] = wt * mat[][] * in0[] * in1[]
template<typename idx_t, typename data_t, typename calc_t, int dof>
inline void matvec_mul(const data_t * mat, const calc_t * in0, const calc_t in1, calc_t * out, calc_t wt) {
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++) {
        calc_t tmp = 0.0;
        const data_t * vals = mat + r * dof;
        #pragma GCC unroll (4)
        for (idx_t c = 0; c < dof; c++)
            tmp += vals[c] * in0[c] * in1[c];
        out[r] = wt * tmp;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
inline void vecvec_mul(const data_t * coeff, calc_t * res) {
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++)
        res[r] *= coeff[r];
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
inline void vecvec_mla(const data_t * vec, const calc_t * in, calc_t * out) {
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++)
        out[r] += vec[r] * in[r];
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
inline void vecvec_mla(const data_t * vec0, const data_t * vec1, const calc_t * in, calc_t * out) {
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++)
        out[r] += vec0[r] * vec1[r] * in[r];
}

template<typename idx_t, typename data_t, int dof>
inline void matmat_mul(const data_t * a, const data_t * b, data_t * res) {
    for (idx_t r = 0; r < dof; r++)
    for (idx_t c = 0; c < dof; c++) {
        data_t tmp = 0.0;
        #pragma GCC unroll (4)
        for (idx_t k = 0; k < dof; k++)
            tmp += a[r * dof + k] * b[k * dof + c];
        res[r * dof + c] = tmp;
    }
}

template<typename idx_t, typename data_t, int dof>
inline void matinv_row(data_t * A, data_t * E) {
    // 构造单位阵
    memset(E, 0.0, sizeof(data_t) * dof * dof);
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++)
        E[r * dof + r] = 1.0;
    // 初等行变换
    for (idx_t i = 0; i < dof; i++) {
        data_t tmp = A[i * dof + i];
        #pragma GCC unroll (4)
        for (idx_t j = 0; j < dof; j++) {
            A[i * dof + j] /= tmp;
            E[i * dof + j] /= tmp;
        }
        // 此时A(i,i)为1，利用这个1消掉上面、下面的行中该列的非零元
        for (idx_t k = 0; k < dof; k++) {
            if (k == i) continue;
            data_t tmp = A[k * dof + i];
            #pragma GCC unroll (4)
            for (idx_t j = 0; j < dof; j++) {
                A[k * dof + j] -= tmp * A[i * dof + j];
                E[k * dof + j] -= tmp * E[i * dof + j];
            }
        }
    }
}

template<typename idx_t, typename data_t, int dof>
inline data_t calc_det_3x3(const data_t * A) {
    data_t tmp0 = A[0] * (A[4] * A[8] - A[5] * A[7]);
    data_t tmp1 = A[1] * (A[3] * A[8] - A[5] * A[6]);
    data_t tmp2 = A[2] * (A[3] * A[7] - A[4] * A[6]);
    return tmp0 - tmp1 + tmp2;
}

// ==========================================================
//  ==================== Normal Kernels ====================
// ==========================================================
template<typename idx_t, typename data_t, typename calc_t, int dof, int num_diag>
void AOS_spmv_3d_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size,
    const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const calc_t * dummy)
{
    constexpr idx_t elms = dof*dof;
    const calc_t * xa[num_diag];
    if constexpr (num_diag == 15) {
        xa[7]  = x_jik;
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;
        xa[11] = x_jik  + vec_dki_size; xa[9]  = x_jik  + vec_dk_size;
        xa[6]  = x_jik  - dof         ; xa[8]  = x_jik  + dof;
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
        xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;
        xa[0]  = xa[3] - vec_dk_size - dof;
        xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;
        xa[14] = xa[11]+ vec_dk_size + dof;
    } else if constexpr (num_diag == 27) {
        xa[13] = x_jik;
        xa[ 4] = x_jik  - vec_dki_size; xa[22] = x_jik  + vec_dki_size;
        xa[10] = x_jik  - vec_dk_size ; xa[16] = x_jik  + vec_dk_size ;
        xa[12] = x_jik  - dof         ; xa[14] = x_jik  + dof         ;
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;
        xa[19] = xa[22]-vec_dk_size ; xa[25] = xa[22]+vec_dk_size ;
        xa[21] = xa[22]-dof         ; xa[23] = xa[22]+dof         ;
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;
        xa[15] = xa[16]-dof         ; xa[17] = xa[16]+dof         ;
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;
        xa[18] = xa[19]-dof         ; xa[20] = xa[19]+dof         ;
        xa[24] = xa[25]-dof         ; xa[26] = xa[25]+dof         ;
    } else if constexpr (num_diag == 7) {
        xa[3] = x_jik;
        xa[0] = x_jik - vec_dki_size; xa[6] = x_jik + vec_dki_size;
        xa[1] = x_jik - vec_dk_size ; xa[5] = x_jik + vec_dk_size ;
        xa[2] = x_jik - dof         ; xa[4] = x_jik + dof         ;
    }
    for (idx_t k = 0; k < num; k++) {
        memset(y_jik, 0, sizeof(calc_t) * dof);
        for (idx_t d = 0; d < num_diag; d++) {
            matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + d*elms, xa[d], y_jik);
            xa[d] += dof;
        }
        A_jik += num_diag * elms;
        y_jik += dof;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag>
void AOS_point_forward_zero_3d_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * dummy0, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy1)
{
    constexpr idx_t elms = dof*dof;
    const calc_t * xa[num_L_diag];
    if constexpr (num_L_diag == 7) {// 3d15
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;
        xa[6]  = x_jik  - dof         ;
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
        xa[4]  = xa[5] - dof       ;
        xa[0]  = xa[3] - vec_dk_size - dof;
    } else if constexpr (num_L_diag == 13) {// 3d27
        xa[ 4] = x_jik  - vec_dki_size;
        xa[10] = x_jik  - vec_dk_size ;
        xa[12] = x_jik  - dof         ;
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;
    } else if constexpr (num_L_diag == 3) {// 3d7
        xa[0] = x_jik - vec_dki_size; xa[1] = x_jik - vec_dk_size ; xa[2] = x_jik - dof         ;
    }
    calc_t tmp[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f];
        for (idx_t d = 0; d < num_L_diag; d++) {
            matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*elms, xa[d], tmp);
            xa[d] += dof;
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, x_jik, - wgt);

        x_jik += dof; b_jik += dof;
        L_jik += num_L_diag * elms;
        invD_jik += elms;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag, int num_U_diag>
void AOS_point_forward_ALL_3d_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy1)
{
    constexpr int elms = dof*dof;
    const calc_t * xa[num_L_diag + 1 + num_U_diag];
    static_assert(num_L_diag == num_U_diag);
    if constexpr (num_L_diag == 7) {// 3d15
        xa[7]  = x_jik;
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;
        xa[11] = x_jik  + vec_dki_size; xa[9]  = x_jik  + vec_dk_size;
        xa[6]  = x_jik  - dof         ; xa[8]  = x_jik  + dof;
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
        xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;
        xa[0]  = xa[3] - vec_dk_size - dof;
        xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;
        xa[14] = xa[11]+ vec_dk_size + dof;
    } else if constexpr (num_L_diag == 13) {// 3d27
        xa[13] = x_jik;
        xa[ 4] = x_jik  - vec_dki_size; xa[22] = x_jik  + vec_dki_size;
        xa[10] = x_jik  - vec_dk_size ; xa[16] = x_jik  + vec_dk_size ;
        xa[12] = x_jik  - dof         ; xa[14] = x_jik  + dof         ;
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;
        xa[19] = xa[22]-vec_dk_size ; xa[25] = xa[22]+vec_dk_size ;
        xa[21] = xa[22]-dof         ; xa[23] = xa[22]+dof         ;
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;
        xa[15] = xa[16]-dof         ; xa[17] = xa[16]+dof         ;
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;
        xa[18] = xa[19]-dof         ; xa[20] = xa[19]+dof         ;
        xa[24] = xa[25]-dof         ; xa[26] = xa[25]+dof         ;
    } else if constexpr (num_L_diag == 3) {// 3d7
        xa[3] = x_jik;
        xa[0] = x_jik - vec_dki_size; xa[6] = x_jik + vec_dki_size;
        xa[1] = x_jik - vec_dk_size ; xa[5] = x_jik + vec_dk_size ;
        xa[2] = x_jik - dof         ; xa[4] = x_jik + dof         ;
    }
    const calc_t one_minus_weight = 1.0 - wgt;
    calc_t tmp[dof], tmp2[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f];

        for (idx_t d = 0; d < num_L_diag; d++) {
            matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*elms, xa[d], tmp);
            xa[d] += dof;
        }
        for (idx_t d = num_L_diag + 1; d < num_L_diag + 1 + num_U_diag; d++) {
            const idx_t p = d - num_L_diag - 1;
            matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + p*elms, xa[d], tmp);
            xa[d] += dof;
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, tmp2, - wgt);
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            x_jik[f] = one_minus_weight * x_jik[f] + tmp2[f];

        x_jik += dof; b_jik += dof;
        L_jik += num_L_diag * elms; U_jik += num_U_diag * elms;
        invD_jik += elms;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag, int num_U_diag>
void AOS_point_backward_ALL_3d_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy1)
{
    constexpr int elms = dof*dof;
    const calc_t * xa[num_L_diag + 1 + num_U_diag];
    static_assert(num_L_diag == num_U_diag);
    if constexpr (num_L_diag == 7) {// 3d15
        xa[7]  = x_jik;
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;
        xa[11] = x_jik  + vec_dki_size; xa[9]  = x_jik  + vec_dk_size;
        xa[6]  = x_jik  - dof         ; xa[8]  = x_jik  + dof;
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
        xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;
        xa[0]  = xa[3] - vec_dk_size - dof;
        xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;
        xa[14] = xa[11]+ vec_dk_size + dof;
    } else if constexpr (num_L_diag == 13) {// 3d27
        xa[13] = x_jik;
        xa[ 4] = x_jik  - vec_dki_size; xa[22] = x_jik  + vec_dki_size;
        xa[10] = x_jik  - vec_dk_size ; xa[16] = x_jik  + vec_dk_size ;
        xa[12] = x_jik  - dof         ; xa[14] = x_jik  + dof         ;
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;
        xa[19] = xa[22]-vec_dk_size ; xa[25] = xa[22]+vec_dk_size ;
        xa[21] = xa[22]-dof         ; xa[23] = xa[22]+dof         ;
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;
        xa[15] = xa[16]-dof         ; xa[17] = xa[16]+dof         ;
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;
        xa[18] = xa[19]-dof         ; xa[20] = xa[19]+dof         ;
        xa[24] = xa[25]-dof         ; xa[26] = xa[25]+dof         ;
    } else if constexpr (num_L_diag == 3) {// 3d7
        xa[3] = x_jik;
        xa[0] = x_jik - vec_dki_size; xa[6] = x_jik + vec_dki_size;
        xa[1] = x_jik - vec_dk_size ; xa[5] = x_jik + vec_dk_size ;
        xa[2] = x_jik - dof         ; xa[4] = x_jik + dof         ;
    }
    const calc_t one_minus_weight = 1.0 - wgt;
    calc_t tmp[dof], tmp2[dof];
    for (idx_t k = 0; k < num; k++) {
        b_jik -= dof; x_jik -= dof;
        L_jik -= num_L_diag * elms; U_jik -= num_U_diag * elms;
        invD_jik -= elms;

        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f];

        for (idx_t d = 0; d < num_L_diag; d++) {
            xa[d] -= dof;
            matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*elms, xa[d], tmp);
        }
        for (idx_t d = num_L_diag + 1; d < num_L_diag + 1 + num_U_diag; d++) {
            xa[d] -= dof;
            const idx_t p = d - num_L_diag - 1;
            matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + p*elms, xa[d], tmp);
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, tmp2, - wgt);
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            x_jik[f] = one_minus_weight * x_jik[f] + tmp2[f];
    }
}


// ==========================================================
//  ==================== Vectorized Kernels ====================
// ==========================================================

// =================================== SPMV ================================

template<int dof, int num_diag>
void AOS_spmv_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size,
    const __fp16 * A_jik, const float * x_jik, float * y_jik, const float * dummy)
{
    const float * xa[num_diag];
    if constexpr (num_diag == 15) {
        xa[7]  = x_jik;
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;
        xa[11] = x_jik  + vec_dki_size; xa[9]  = x_jik  + vec_dk_size;
        xa[6]  = x_jik  - dof         ; xa[8]  = x_jik  + dof;
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
        xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;
        xa[0]  = xa[3] - vec_dk_size - dof;
        xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;
        xa[14] = xa[11]+ vec_dk_size + dof;
    } else if constexpr (num_diag == 27) {
        xa[13] = x_jik;
        xa[ 4] = x_jik  - vec_dki_size; xa[22] = x_jik  + vec_dki_size;
        xa[10] = x_jik  - vec_dk_size ; xa[16] = x_jik  + vec_dk_size ;
        xa[12] = x_jik  - dof         ; xa[14] = x_jik  + dof         ;
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;
        xa[19] = xa[22]-vec_dk_size ; xa[25] = xa[22]+vec_dk_size ;
        xa[21] = xa[22]-dof         ; xa[23] = xa[22]+dof         ;
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;
        xa[15] = xa[16]-dof         ; xa[17] = xa[16]+dof         ;
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;
        xa[18] = xa[19]-dof         ; xa[20] = xa[19]+dof         ;
        xa[24] = xa[25]-dof         ; xa[26] = xa[25]+dof         ;
    } else if constexpr (num_diag == 7) {
        xa[3] = x_jik;
        xa[0] = x_jik - vec_dki_size; xa[6] = x_jik + vec_dki_size;
        xa[1] = x_jik - vec_dk_size ; xa[5] = x_jik + vec_dk_size ;
        xa[2] = x_jik - dof         ; xa[4] = x_jik + dof         ;
    }
    constexpr int elms = dof*dof;
    if constexpr (dof == 3) {
        float16x4x3_t   A0_2_16;
        float32x4_t c0, c1, c2;
        int k = 0, max_3k = num & (~2);
        constexpr int mat_prft = elms * num_diag;
        for ( ; k < max_3k; k += 3) {
            const __fp16 * aos_ptr = A_jik;
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0),// 第几列的结果暂存
                        s0 = vdupq_n_f32(0.0), s1 = vdupq_n_f32(0.0), s2 = vdupq_n_f32(0.0),
                        t0 = vdupq_n_f32(0.0), t1 = vdupq_n_f32(0.0), t2 = vdupq_n_f32(0.0);
            float32x4_t b0, b1, b2, a0, a1, a2;
            for (int d = 0; d < num_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 3*mat_prft, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);
                // 15 * 9 = 135
                A0_2_16 = vld3_f16(aos_ptr + mat_prft); __builtin_prefetch(aos_ptr + 4*mat_prft, 0, 0);
                b0 = vcvt_f32_f16(A0_2_16.val[0]); b1 = vcvt_f32_f16(A0_2_16.val[1]); b2 = vcvt_f32_f16(A0_2_16.val[2]);
                s0 = vmlaq_n_f32(s0, b0, xp[0])  ; s1 = vmlaq_n_f32(s1, b1, xp[1])  ; s2 = vmlaq_n_f32(s2, b2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);

                A0_2_16 = vld3_f16(aos_ptr + 2*mat_prft); __builtin_prefetch(aos_ptr + 5*mat_prft, 0, 0);
                a0 = vcvt_f32_f16(A0_2_16.val[0]); a1 = vcvt_f32_f16(A0_2_16.val[1]); a2 = vcvt_f32_f16(A0_2_16.val[2]);
                t0 = vmlaq_n_f32(t0, a0, xp[0])  ; t1 = vmlaq_n_f32(t1, a1, xp[1])  ; t2 = vmlaq_n_f32(t2, a2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);

                aos_ptr += elms;
            }
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            s0 = vaddq_f32(s0, s1); s0 = vaddq_f32(s0, s2);
            t0 = vaddq_f32(t0, t1); t0 = vaddq_f32(t0, t2);
            vst1q_f32(y_jik, r0); y_jik += dof; __builtin_prefetch(y_jik + dof,1);
            vst1q_f32(y_jik, s0); y_jik += dof; __builtin_prefetch(y_jik + dof,1);
            vst1q_f32(y_jik, t0); y_jik += dof; __builtin_prefetch(y_jik + dof,1);
            
            A_jik += 3*num_diag * elms;// 15 * 3 = 45
        }
        for (k = 0; k < num - max_3k; k++) {// 做完剩下的元素
            const __fp16 * aos_ptr = A_jik; 
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            for (int d = 0; d < num_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + mat_prft, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ; 
                xp += dof; __builtin_prefetch(xp + dof, 0);

                aos_ptr += elms;
            }
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            vst1q_f32(y_jik, r0); y_jik += dof; __builtin_prefetch(y_jik + dof,1);
            A_jik += num_diag * elms;
        }
    }
}


// ================================= PGS ====================================


template<int dof, int num_L_diag>
void AOS_point_forward_zero_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * dummy0, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * dummy1)
{
    constexpr int elms = dof*dof;
    const float * xa[num_L_diag];
    if constexpr (num_L_diag == 7) {// 3d15
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;
        xa[6]  = x_jik  - dof         ;
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
        xa[4]  = xa[5] - dof       ;
        xa[0]  = xa[3] - vec_dk_size - dof;
    } else if constexpr (num_L_diag == 13) {// 3d27
        xa[ 4] = x_jik  - vec_dki_size;
        xa[10] = x_jik  - vec_dk_size ;
        xa[12] = x_jik  - dof         ;
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;
    } else if constexpr (num_L_diag == 3) {// 3d7
        xa[0] = x_jik - vec_dki_size; xa[1] = x_jik - vec_dk_size ; xa[2] = x_jik - dof         ;
    }
    const float32x4_t vwgts = vdupq_n_f32(wgt);
    if constexpr (dof == 3) {
        float16x4x3_t A0_2_16;
        float32x4_t c0, c1, c2, tmp0;
        for (int k = 0; k < num; k++) {// 做完剩下的元素
            const __fp16 * aos_ptr = L_jik; 
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            tmp0 = vld1q_f32(b_jik); b_jik += dof; __builtin_prefetch(b_jik + 2*dof, 0, 0);
            
            for (int d = 0; d < num_L_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 2*elms * num_L_diag, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ; 
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            L_jik += num_L_diag * elms;

            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            tmp0 = vsubq_f32(tmp0, r0);// 此时tmp暂存了 b - 非本位的aj*xj
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            A0_2_16 = vld3_f16(invD_jik); invD_jik += elms; __builtin_prefetch(invD_jik + 2*elms, 0, 0);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            float buf[4]; vst1q_f32(buf, r0);// 不能直接存，必须先倒一趟，以免向量寄存器内的lane3会污染原数据
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x_jik[f] = buf[f];

            x_jik += dof; __builtin_prefetch(x_jik + 2*dof,1);
        }
    }
}

template<int dof, int num_L_diag, int num_U_diag>
void AOS_point_forward_ALL_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * U_jik, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * dummy)
{
    constexpr int elms = dof*dof;
    const float * xa[num_L_diag + 1 + num_U_diag];
    static_assert(num_L_diag == num_U_diag);
    if constexpr (num_L_diag == 7) {// 3d15
        xa[7]  = x_jik;
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;
        xa[11] = x_jik  + vec_dki_size; xa[9]  = x_jik  + vec_dk_size;
        xa[6]  = x_jik  - dof         ; xa[8]  = x_jik  + dof;
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
        xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;
        xa[0]  = xa[3] - vec_dk_size - dof;
        xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;
        xa[14] = xa[11]+ vec_dk_size + dof;
    } else if constexpr (num_L_diag == 13) {// 3d27
        xa[13] = x_jik;
        xa[ 4] = x_jik  - vec_dki_size; xa[22] = x_jik  + vec_dki_size;
        xa[10] = x_jik  - vec_dk_size ; xa[16] = x_jik  + vec_dk_size ;
        xa[12] = x_jik  - dof         ; xa[14] = x_jik  + dof         ;
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;
        xa[19] = xa[22]-vec_dk_size ; xa[25] = xa[22]+vec_dk_size ;
        xa[21] = xa[22]-dof         ; xa[23] = xa[22]+dof         ;
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;
        xa[15] = xa[16]-dof         ; xa[17] = xa[16]+dof         ;
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;
        xa[18] = xa[19]-dof         ; xa[20] = xa[19]+dof         ;
        xa[24] = xa[25]-dof         ; xa[26] = xa[25]+dof         ;
    } else if constexpr (num_L_diag == 3) {// 3d7
        xa[3] = x_jik;
        xa[0] = x_jik - vec_dki_size; xa[6] = x_jik + vec_dki_size;
        xa[1] = x_jik - vec_dk_size ; xa[5] = x_jik + vec_dk_size ;
        xa[2] = x_jik - dof         ; xa[4] = x_jik + dof         ;
    }
    const float32x4_t vwgts = vdupq_n_f32(wgt), vone_minus_wgts = vdupq_n_f32(1.0 - wgt);
    if constexpr (dof == 3) {
        const __fp16 * aos_ptr = nullptr;
        float16x4x3_t A0_2_16;
        float32x4_t c0, c1, c2, tmp0;

        for (int k = 0; k < num; k++) {
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            tmp0 = vld1q_f32(b_jik); b_jik += dof; __builtin_prefetch(b_jik + 2*dof, 0, 0);

            aos_ptr = L_jik;// L部分
            for (int d = 0; d < num_L_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 2*elms * num_L_diag, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ; 
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            L_jik += num_L_diag * elms;

            aos_ptr = U_jik;// U部分
            for (int d = num_L_diag + 1; d < num_L_diag + 1 + num_U_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 2*elms * num_U_diag, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ; 
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            U_jik += num_U_diag * elms;

            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            tmp0 = vsubq_f32(tmp0, r0);// 此时tmp暂存了 b - 非本位的aj*xj
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            A0_2_16 = vld3_f16(invD_jik); invD_jik += elms; __builtin_prefetch(invD_jik + 2*elms, 0, 0);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            tmp0 = vld1q_f32(x_jik);// 此时tmp0暂存原来的x解
            r0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w*D^{-1}*(b - 非本位的aj*xj)
            float buf[4]; vst1q_f32(buf, r0);
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x_jik[f] = buf[f];
            
            // printf(" k %d res %.6e %.6e %.6e\n", k, x7[0], x7[1], x7[2]);
            x_jik += dof; __builtin_prefetch(x_jik + 2*dof,1);
        }
    }
}


template<int dof, int num_L_diag, int num_U_diag>
void AOS_point_backward_ALL_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * U_jik, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * dummy)
{
    constexpr int elms = dof*dof;
    const float * xa[num_L_diag + 1 + num_U_diag];
    static_assert(num_L_diag == num_U_diag);
    if constexpr (num_L_diag == 7) {// 3d15
        xa[7]  = x_jik;
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;
        xa[11] = x_jik  + vec_dki_size; xa[9]  = x_jik  + vec_dk_size;
        xa[6]  = x_jik  - dof         ; xa[8]  = x_jik  + dof;
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
        xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;
        xa[0]  = xa[3] - vec_dk_size - dof;
        xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;
        xa[14] = xa[11]+ vec_dk_size + dof;
    } else if constexpr (num_L_diag == 13) {// 3d27
        xa[13] = x_jik;
        xa[ 4] = x_jik  - vec_dki_size; xa[22] = x_jik  + vec_dki_size;
        xa[10] = x_jik  - vec_dk_size ; xa[16] = x_jik  + vec_dk_size ;
        xa[12] = x_jik  - dof         ; xa[14] = x_jik  + dof         ;
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;
        xa[19] = xa[22]-vec_dk_size ; xa[25] = xa[22]+vec_dk_size ;
        xa[21] = xa[22]-dof         ; xa[23] = xa[22]+dof         ;
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;
        xa[15] = xa[16]-dof         ; xa[17] = xa[16]+dof         ;
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;
        xa[18] = xa[19]-dof         ; xa[20] = xa[19]+dof         ;
        xa[24] = xa[25]-dof         ; xa[26] = xa[25]+dof         ;
    } else if constexpr (num_L_diag == 3) {// 3d7
        xa[3] = x_jik;
        xa[0] = x_jik - vec_dki_size; xa[6] = x_jik + vec_dki_size;
        xa[1] = x_jik - vec_dk_size ; xa[5] = x_jik + vec_dk_size ;
        xa[2] = x_jik - dof         ; xa[4] = x_jik + dof         ;
    }
    const float32x4_t vwgts = vdupq_n_f32(wgt), vone_minus_wgts = vdupq_n_f32(1.0 - wgt);
    if constexpr (dof == 3) {
        const __fp16 * aos_ptr = nullptr;
        float16x4x3_t A0_2_16;
        float32x4_t c0, c1, c2, tmp0;

        for (int k = 0; k < num; k++) {// 做完剩下的元素
            b_jik -= dof; __builtin_prefetch(b_jik - 2*dof, 0, 0);
            float32x4_t r0 = vld1q_f32(b_jik), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存

            L_jik -= num_L_diag * elms; aos_ptr = L_jik;
            for (int d = 0; d < num_L_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr - 2*elms * num_L_diag, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                xp -= dof; __builtin_prefetch(xp - 2*dof, 0);
                r0 = vmlsq_n_f32(r0, c0, xp[0])  ; r1 = vmlsq_n_f32(r1, c1, xp[1])  ; r2 = vmlsq_n_f32(r2, c2, xp[2])  ; 
                aos_ptr += elms;
            }

            U_jik -= num_U_diag * elms; aos_ptr = U_jik;
            for (int d = num_L_diag + 1; d < num_L_diag + 1 + num_U_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr - 2*elms * num_U_diag, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                xp -= dof; __builtin_prefetch(xp - 2*dof, 0);
                r0 = vmlsq_n_f32(r0, c0, xp[0])  ; r1 = vmlsq_n_f32(r1, c1, xp[1])  ; r2 = vmlsq_n_f32(r2, c2, xp[2])  ; 
                aos_ptr += elms;
            }

            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            tmp0 = vmulq_f32(r0, vwgts);// 此时tmp暂存了 w*(b - 非本位的aj*xj)

            invD_jik -= elms;
            A0_2_16 = vld3_f16(invD_jik); __builtin_prefetch(invD_jik - 2*elms, 0, 0);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);

            x_jik -= dof; __builtin_prefetch(x_jik - 2*dof,1);
            tmp0 = vld1q_f32(x_jik);// 此时tmp0暂存原来的x解
            tmp0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w*D^{-1}*(b - 非本位的aj*xj)

            float buf[4]; vst1q_f32(buf, tmp0);
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x_jik[f] = buf[f];
        }
    }
}

#endif