#ifndef SYSSMG_SCAL_KERNELS_HPP
#define SYSSMG_SCAL_KERNELS_HPP

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
inline void matvec_mla(const data_t * mat, const calc_t * vec, const calc_t * in, calc_t * out) {
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

template<typename idx_t, typename calc_t, int dof>
inline void vecvec_mul(const calc_t * coeff, calc_t * res) {
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++)
        res[r] *= coeff[r];
}

template<typename idx_t, typename calc_t, int dof>
inline void vecvec_div(calc_t * a, const calc_t * b) {
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++)
        a[r] /= b[r];
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
inline void vecvec_mla(const data_t * vec, const calc_t * in, calc_t * out) {
    #pragma GCC unroll (4)
    for (idx_t r = 0; r < dof; r++)
        out[r] += vec[r] * in[r];
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
inline void vecvec_mla(const data_t * vec0, const calc_t * vec1, const calc_t * in, calc_t * out) {
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

//  ------------------ SPMV 
#define spmv_prepare_xa \
    if constexpr (num_diag == 15) {\
        xa[7]  = x_jik;\
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;\
        xa[11] = x_jik  + vec_dki_size; xa[9]  = x_jik  + vec_dk_size;\
        xa[6]  = x_jik  - dof         ; xa[8]  = x_jik  + dof;\
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;\
        xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;\
        xa[0]  = xa[3] - vec_dk_size - dof;\
        xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;\
        xa[14] = xa[11]+ vec_dk_size + dof;\
    } else if constexpr (num_diag == 27) {\
        xa[13] = x_jik;\
        xa[ 4] = x_jik  - vec_dki_size; xa[22] = x_jik  + vec_dki_size;\
        xa[10] = x_jik  - vec_dk_size ; xa[16] = x_jik  + vec_dk_size ;\
        xa[12] = x_jik  - dof         ; xa[14] = x_jik  + dof         ;\
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;\
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;\
        xa[19] = xa[22]-vec_dk_size ; xa[25] = xa[22]+vec_dk_size ;\
        xa[21] = xa[22]-dof         ; xa[23] = xa[22]+dof         ;\
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;\
        xa[15] = xa[16]-dof         ; xa[17] = xa[16]+dof         ;\
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;\
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;\
        xa[18] = xa[19]-dof         ; xa[20] = xa[19]+dof         ;\
        xa[24] = xa[25]-dof         ; xa[26] = xa[25]+dof         ;\
    } else if constexpr (num_diag == 7) {\
        xa[3] = x_jik;\
        xa[0] = x_jik - vec_dki_size; xa[6] = x_jik + vec_dki_size;\
        xa[1] = x_jik - vec_dk_size ; xa[5] = x_jik + vec_dk_size ;\
        xa[2] = x_jik - dof         ; xa[4] = x_jik + dof         ;\
    }
    
template<typename idx_t, typename data_t, typename calc_t, int dof, int num_diag>
void AOS_spmv_3d_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size,
    const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const calc_t * dummy)
{
    constexpr idx_t elms = dof*dof;
    const calc_t * xa[num_diag];
    spmv_prepare_xa;
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

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_diag>
void AOS_spmv_3d_scaled_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size,
    const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const calc_t * sqD_jik)
{
    constexpr idx_t elms = dof*dof;
    const calc_t * xa[num_diag], * sqDa[num_diag];
    spmv_prepare_xa;
    for (idx_t d = 0; d < num_diag; d++)
        sqDa[d] = sqD_jik + (xa[d] - x_jik);// 两者向量的偏移是一样的
    for (idx_t k = 0; k < num; k++) {
        memset(y_jik, 0, sizeof(calc_t) * dof);
        for (idx_t d = 0; d < num_diag; d++) {
            matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + d*elms, sqDa[d], xa[d], y_jik);
            xa[d] += dof;
            sqDa[d] += dof;
        }
        vecvec_mul<idx_t, calc_t, dof>(sqD_jik, y_jik);

        A_jik += num_diag * elms;
        y_jik += dof; sqD_jik += dof;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag, int num_U_diag>
void AOS_compress_spmv_3d_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size,
    const data_t * L_jik, const data_t * D_jik, const data_t * U_jik,
    const calc_t * x_jik, calc_t * y_jik, const calc_t * dummy)
{
    static_assert(num_L_diag == num_U_diag);
    if constexpr (num_L_diag == 3) {
        const calc_t*x3 = x_jik,
                    *x0 = x_jik - vec_dki_size, *x6 = x_jik + vec_dki_size,
                    *x1 = x_jik - vec_dk_size , *x5 = x_jik + vec_dk_size ,
                    *x2 = x_jik - dof         , *x4 = x_jik + dof         ;
        for (idx_t k = 0; k < num; k++) {
            memset(y_jik, 0, sizeof(calc_t) * dof);
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik        , x0, y_jik); x0 += dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik +   dof, x1, y_jik); x1 += dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 2*dof, x2, y_jik); x2 += dof; L_jik += 3*dof;
            matvec_mla<idx_t, data_t, calc_t, dof>(D_jik        , x3, y_jik); x3 += dof; D_jik += dof*dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik        , x4, y_jik); x4 += dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik +   dof, x5, y_jik); x5 += dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 2*dof, x6, y_jik); x6 += dof; U_jik += 3*dof;
            y_jik += dof;
        }
    }
    else {
        constexpr idx_t num_diag = num_L_diag + 1 + num_U_diag;
        static_assert(num_diag >> 1 == num_L_diag);
        const calc_t * xa[num_diag];
        spmv_prepare_xa;
        constexpr idx_t diag_id = num_L_diag;
        for (idx_t k = 0; k < num; k++) {
            memset(y_jik, 0, sizeof(calc_t) * dof);
            for (idx_t d = 0; d < diag_id; d++) {
                vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*dof, xa[d], y_jik);
                xa[d] += dof;
            }
            
            matvec_mla<idx_t, data_t, calc_t, dof>(D_jik, xa[diag_id], y_jik);
            xa[diag_id] += dof;

            for (idx_t d = diag_id + 1; d < num_diag; d++) {
                const idx_t p = d - num_L_diag - 1;
                vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik + p*dof, xa[d], y_jik);
                xa[d] += dof;
            }

            L_jik += num_L_diag * dof; D_jik += dof*dof; U_jik += num_U_diag * dof;
            y_jik += dof;
        }
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag, int num_U_diag>
void AOS_compress_spmv_3d_scaled_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size,
    const data_t * L_jik, const data_t * D_jik, const data_t * U_jik,
    const calc_t * x_jik, calc_t * y_jik, const calc_t * sqD_jik)
{
    static_assert(num_L_diag == num_U_diag);
    if constexpr (num_L_diag == 3) {
        const calc_t*x3 = x_jik,
                    *x0 = x_jik - vec_dki_size, *x6 = x_jik + vec_dki_size,
                    *x1 = x_jik - vec_dk_size , *x5 = x_jik + vec_dk_size ,
                    *x2 = x_jik - dof         , *x4 = x_jik + dof         ;
        const calc_t* sqD3 = sqD_jik,
                    * sqD0 = sqD_jik - vec_dki_size, *sqD6 = sqD_jik + vec_dki_size,
                    * sqD1 = sqD_jik - vec_dk_size , *sqD5 = sqD_jik + vec_dk_size ,
                    * sqD2 = sqD_jik - dof         , *sqD4 = sqD_jik + dof         ;
        for (idx_t k = 0; k < num; k++) {
            memset(y_jik, 0, sizeof(calc_t) * dof);
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik        , sqD0, x0, y_jik); sqD0 += dof; x0 += dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik +   dof, sqD1, x1, y_jik); sqD1 += dof; x1 += dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 2*dof, sqD2, x2, y_jik); sqD2 += dof; x2 += dof; L_jik += 3*dof;
            matvec_mla<idx_t, data_t, calc_t, dof>(D_jik        , sqD3, x3, y_jik); sqD3 += dof; x3 += dof; D_jik += dof*dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik        , sqD4, x4, y_jik); sqD4 += dof; x4 += dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik +   dof, sqD5, x5, y_jik); sqD5 += dof; x5 += dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 2*dof, sqD6, x6, y_jik); sqD6 += dof; x6 += dof; U_jik += 3*dof;
            vecvec_mul<idx_t, calc_t, dof>(sqD_jik, y_jik); sqD_jik += dof; y_jik += dof;
        }
    }
    else {
        constexpr idx_t num_diag = num_L_diag + 1 + num_U_diag;
        static_assert(num_diag >> 1 == num_L_diag);
        const calc_t * xa[num_diag], * sqDa[num_diag];
        spmv_prepare_xa;
        for (idx_t d = 0; d < num_diag; d++)
            sqDa[d] = sqD_jik + (xa[d] - x_jik);// 两者向量的偏移是一样的
        constexpr idx_t diag_id = num_L_diag;
        for (idx_t k = 0; k < num; k++) {
            memset(y_jik, 0, sizeof(calc_t) * dof);
            for (idx_t d = 0; d < diag_id; d++) {
                vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*dof, sqDa[d], xa[d], y_jik);
                xa[d] += dof;
                sqDa[d] +=dof;
            }

            for (idx_t d = diag_id + 1; d < num_diag; d++) {
                const idx_t p = d - num_L_diag - 1;
                vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik + p*dof, sqDa[d], xa[d], y_jik);
                xa[d] += dof;
                sqDa[d] += dof;
            }

            matvec_mla<idx_t, data_t, calc_t, dof>(D_jik, sqDa[diag_id], xa[diag_id], y_jik);
            xa[diag_id] += dof;
            sqDa[diag_id] += dof;

            vecvec_mul<idx_t, calc_t, dof>(sqD_jik, y_jik);

            L_jik += num_L_diag * dof; D_jik += dof*dof; U_jik += num_U_diag * dof;
            y_jik += dof; sqD_jik += dof;
        }
    }
}


// -------------------- PGS

#define pgsf_prepare_xa \
    if constexpr (num_L_diag == 7) {\
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;\
        xa[6]  = x_jik  - dof         ;\
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;\
        xa[4]  = xa[5] - dof       ;\
        xa[0]  = xa[3] - vec_dk_size - dof;\
    } else if constexpr (num_L_diag == 13) {\
        xa[ 4] = x_jik  - vec_dki_size;\
        xa[10] = x_jik  - vec_dk_size ;\
        xa[12] = x_jik  - dof         ;\
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;\
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;\
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;\
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;\
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;\
    } else if constexpr (num_L_diag == 3) {\
        xa[0] = x_jik - vec_dki_size; xa[1] = x_jik - vec_dk_size ; xa[2] = x_jik - dof         ;\
    }

#define pgsAll_prepare_xa \
    if constexpr (num_L_diag == 7) {\
        xa[7]  = x_jik;\
        xa[3]  = x_jik  - vec_dki_size; xa[5]  = x_jik  - vec_dk_size;\
        xa[11] = x_jik  + vec_dki_size; xa[9]  = x_jik  + vec_dk_size;\
        xa[6]  = x_jik  - dof         ; xa[8]  = x_jik  + dof;\
        xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;\
        xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;\
        xa[0]  = xa[3] - vec_dk_size - dof;\
        xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;\
        xa[14] = xa[11]+ vec_dk_size + dof;\
    } else if constexpr (num_L_diag == 13) {\
        xa[13] = x_jik;\
        xa[ 4] = x_jik  - vec_dki_size; xa[22] = x_jik  + vec_dki_size;\
        xa[10] = x_jik  - vec_dk_size ; xa[16] = x_jik  + vec_dk_size ;\
        xa[12] = x_jik  - dof         ; xa[14] = x_jik  + dof         ;\
        xa[ 1] = xa[4]- vec_dk_size ; xa[ 7] = xa[4]+ vec_dk_size ;\
        xa[ 3] = xa[4]- dof         ; xa[ 5] = xa[4]+ dof         ;\
        xa[19] = xa[22]-vec_dk_size ; xa[25] = xa[22]+vec_dk_size ;\
        xa[21] = xa[22]-dof         ; xa[23] = xa[22]+dof         ;\
        xa[ 9] = xa[10]-dof         ; xa[11] = xa[10]+dof         ;\
        xa[15] = xa[16]-dof         ; xa[17] = xa[16]+dof         ;\
        xa[ 0] = xa[ 1]-dof         ; xa[ 2] = xa[ 1]+dof         ;\
        xa[ 6] = xa[ 7]-dof         ; xa[ 8] = xa[ 7]+dof         ;\
        xa[18] = xa[19]-dof         ; xa[20] = xa[19]+dof         ;\
        xa[24] = xa[25]-dof         ; xa[26] = xa[25]+dof         ;\
    } else if constexpr (num_L_diag == 3) {\
        xa[3] = x_jik;\
        xa[0] = x_jik - vec_dki_size; xa[6] = x_jik + vec_dki_size;\
        xa[1] = x_jik - vec_dk_size ; xa[5] = x_jik + vec_dk_size ;\
        xa[2] = x_jik - dof         ; xa[4] = x_jik + dof         ;\
    }
    
//  - - - - - - - - - - - - - - - 正常的函数

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag>
void AOS_point_forward_zero_3d_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * dummy0, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy1)
{
    constexpr idx_t elms = dof*dof;
    const calc_t * xa[num_L_diag];
    pgsf_prepare_xa;
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
    pgsAll_prepare_xa;
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
    pgsAll_prepare_xa;
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

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag>
void AOS_point_forward_zero_3d_scaled_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * dummy0, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * sqD_jik)
{
    constexpr idx_t elms = dof*dof;
    const calc_t * xa[num_L_diag], * sqDa[num_L_diag];
    pgsf_prepare_xa;
    for (idx_t d = 0; d < num_L_diag; d++)
        sqDa[d] = sqD_jik + (xa[d] - x_jik);
    calc_t tmp[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)// - Q^{-1/2}*b
            tmp[f] = - b_jik[f] / sqD_jik[f];

        for (idx_t d = 0; d < num_L_diag; d++) {// - Q^{-1/2}*b + (Lbar + Ubar)*Q^{1/2}*x
            matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*elms, sqDa[d], xa[d], tmp);
            xa[d] += dof;
            sqDa[d] += dof;
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, x_jik, - wgt);// w*Dbar^{-1}*(Q^{-1/2}*b - (Lbar + Ubar)*Q^{1/2}*x) 
        vecvec_div<idx_t, calc_t, dof>(x_jik, sqD_jik);

        x_jik += dof; b_jik += dof; sqD_jik += dof;
        L_jik += num_L_diag * elms;
        invD_jik += elms;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag, int num_U_diag>
void AOS_point_forward_ALL_3d_scaled_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * sqD_jik)
{
    constexpr int elms = dof*dof;
    constexpr int num_diag = num_L_diag + 1 + num_U_diag;
    const calc_t * xa[num_diag], * sqDa[num_diag];
    static_assert(num_L_diag == num_U_diag);
    pgsAll_prepare_xa;
    for (idx_t d = 0; d < num_diag; d++)
        sqDa[d] = sqD_jik + (xa[d] - x_jik);
    const calc_t one_minus_weight = 1.0 - wgt;
    calc_t tmp[dof], tmp2[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f] / sqD_jik[f];// - Q^{-1/2}*b

        for (idx_t d = 0; d < num_L_diag; d++) {
            matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*elms, sqDa[d], xa[d], tmp);
            xa[d] += dof;
            sqDa[d] += dof;
        }
        for (idx_t d = num_L_diag + 1; d < num_diag; d++) {
            const idx_t p = d - num_L_diag - 1;
            matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + p*elms, sqDa[d], xa[d], tmp);
            xa[d] += dof;
            sqDa[d] += dof;
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, tmp2, - wgt);
        vecvec_div<idx_t, calc_t, dof>(tmp2, sqD_jik);
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            x_jik[f] = one_minus_weight * x_jik[f] + tmp2[f];

        x_jik += dof; b_jik += dof; sqD_jik += dof;
        L_jik += num_L_diag * elms; U_jik += num_U_diag * elms;
        invD_jik += elms;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag, int num_U_diag>
void AOS_point_backward_ALL_3d_scaled_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * sqD_jik)
{
    constexpr int elms = dof*dof;
    constexpr int num_diag = num_L_diag + 1 + num_U_diag;
    const calc_t * xa[num_diag], * sqDa[num_diag];
    static_assert(num_L_diag == num_U_diag);
    pgsAll_prepare_xa;
    for (idx_t d = 0; d < num_diag; d++)
        sqDa[d] = sqD_jik + (xa[d] - x_jik);
    const calc_t one_minus_weight = 1.0 - wgt;
    calc_t tmp[dof], tmp2[dof];
    for (idx_t k = 0; k < num; k++) {
        b_jik -= dof; x_jik -= dof; sqD_jik -= dof;
        L_jik -= num_L_diag * elms; U_jik -= num_U_diag * elms;
        invD_jik -= elms;

        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f] / sqD_jik[f];

        for (idx_t d = 0; d < num_L_diag; d++) {
            xa[d] -= dof;
            sqDa[d] -= dof;
            matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*elms, sqDa[d], xa[d], tmp);
        }
        for (idx_t d = num_L_diag + 1; d < num_diag; d++) {
            xa[d] -= dof;
            sqDa[d] -= dof;
            const idx_t p = d - num_L_diag - 1;
            matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + p*elms, sqDa[d], xa[d], tmp);
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, tmp2, - wgt);
        vecvec_div<idx_t, calc_t, dof>(tmp2, sqD_jik);
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            x_jik[f] = one_minus_weight * x_jik[f] + tmp2[f];
    }
}

//  - - - - - - - - - - - - - - - 带压缩的函数

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag>
void AOS_compress_point_forward_zero_3d_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * dummy0, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy1)
{
    const calc_t * xa[num_L_diag];
    pgsf_prepare_xa;
    calc_t tmp[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f];
        for (idx_t d = 0; d < num_L_diag; d++) {
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*dof, xa[d], tmp);
            xa[d] += dof;
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, x_jik, - wgt);

        x_jik += dof; b_jik += dof;
        L_jik += num_L_diag * dof;
        invD_jik += dof*dof;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag, int num_U_diag>
void AOS_compress_point_forward_ALL_3d_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy1)
{
    constexpr idx_t num_diag = num_L_diag + 1 + num_U_diag;
    const calc_t * xa[num_diag];
    static_assert(num_L_diag == num_U_diag);
    pgsAll_prepare_xa;
    const calc_t one_minus_weight = 1.0 - wgt;
    calc_t tmp[dof], tmp2[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f];

        for (idx_t d = 0; d < num_L_diag; d++) {
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*dof, xa[d], tmp);
            xa[d] += dof;
        }
        for (idx_t d = num_L_diag + 1; d < num_diag; d++) {
            const idx_t p = d - num_L_diag - 1;
            vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik + p*dof, xa[d], tmp);
            xa[d] += dof;
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, tmp2, - wgt);
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            x_jik[f] = one_minus_weight * x_jik[f] + tmp2[f];

        x_jik += dof; b_jik += dof;
        L_jik += num_L_diag * dof; U_jik += num_U_diag * dof;
        invD_jik += dof*dof;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag, int num_U_diag>
void AOS_compress_point_backward_ALL_3d_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy1) 
{
    constexpr idx_t num_diag = num_L_diag + 1 + num_U_diag;
    const calc_t * xa[num_diag];
    static_assert(num_L_diag == num_U_diag);
    pgsAll_prepare_xa;
    const calc_t one_minus_weight = 1.0 - wgt;
    calc_t tmp[dof], tmp2[dof];
    for (idx_t k = 0; k < num; k++) {
        b_jik -= dof; x_jik -= dof;
        L_jik -= num_L_diag * dof; U_jik -= num_U_diag * dof;
        invD_jik -= dof*dof;

        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f];

        for (idx_t d = 0; d < num_L_diag; d++) {
            xa[d] -= dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*dof, xa[d], tmp);
        }
        for (idx_t d = num_L_diag + 1; d < num_diag; d++) {
            xa[d] -= dof;
            const idx_t p = d - num_L_diag - 1;
            vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik + p*dof, xa[d], tmp);
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, tmp2, - wgt);
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            x_jik[f] = one_minus_weight * x_jik[f] + tmp2[f];
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag>
void AOS_compress_point_forward_zero_3d_scaled_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * dummy0, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * sqD_jik)
{
    const calc_t * xa[num_L_diag], * sqDa[num_L_diag];
    pgsf_prepare_xa;
    for (idx_t d = 0; d < num_L_diag; d++)
        sqDa[d] = sqD_jik + (xa[d] - x_jik);
    calc_t tmp[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f] / sqD_jik[f];

        for (idx_t d = 0; d < num_L_diag; d++) {// - Q^{-1/2}*b + (Lbar + Ubar)*Q^{1/2}*x
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*dof, sqDa[d], xa[d], tmp);
            xa[d] += dof;
            sqDa[d] += dof;
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, x_jik, - wgt);// w*Dbar^{-1}*(Q^{-1/2}*b - (Lbar + Ubar)*Q^{1/2}*x) 
        vecvec_div<idx_t, calc_t, dof>(x_jik, sqD_jik);

        x_jik += dof; b_jik += dof; sqD_jik += dof;
        L_jik += num_L_diag * dof;
        invD_jik += dof*dof;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag, int num_U_diag>
void AOS_compress_point_forward_ALL_3d_scaled_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * sqD_jik)
{
    constexpr int num_diag = num_L_diag + 1 + num_U_diag;
    const calc_t * xa[num_diag], * sqDa[num_diag];
    static_assert(num_L_diag == num_U_diag);
    pgsAll_prepare_xa;
    for (idx_t d = 0; d < num_diag; d++)
        sqDa[d] = sqD_jik + (xa[d] - x_jik);
    const calc_t one_minus_weight = 1.0 - wgt;
    calc_t tmp[dof], tmp2[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f] / sqD_jik[f];// - Q^{-1/2}*b

        for (idx_t d = 0; d < num_L_diag; d++) {
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*dof, sqDa[d], xa[d], tmp);
            xa[d] += dof;
            sqDa[d] += dof;
        }
        for (idx_t d = num_L_diag + 1; d < num_diag; d++) {
            const idx_t p = d - num_L_diag - 1;
            vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik + p*dof, sqDa[d], xa[d], tmp);
            xa[d] += dof;
            sqDa[d] += dof;
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, tmp2, - wgt);
        vecvec_div<idx_t, calc_t, dof>(tmp2, sqD_jik);
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            x_jik[f] = one_minus_weight * x_jik[f] + tmp2[f];

        x_jik += dof; b_jik += dof; sqD_jik += dof;
        L_jik += num_L_diag * dof; U_jik += num_U_diag * dof;
        invD_jik += dof*dof;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof, int num_L_diag, int num_U_diag>
void AOS_compress_point_backward_ALL_3d_scaled_normal(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * invD_jik,
    const calc_t * b_jik, calc_t * x_jik, const calc_t * sqD_jik)
{
    constexpr int num_diag = num_L_diag + 1 + num_U_diag;
    const calc_t * xa[num_diag], * sqDa[num_diag];
    static_assert(num_L_diag == num_U_diag);
    pgsAll_prepare_xa;
    for (idx_t d = 0; d < num_diag; d++)
        sqDa[d] = sqD_jik + (xa[d] - x_jik);
    const calc_t one_minus_weight = 1.0 - wgt;
    calc_t tmp[dof], tmp2[dof];
    for (idx_t k = 0; k < num; k++) {
        b_jik -= dof; x_jik -= dof; sqD_jik -= dof;
        L_jik -= num_L_diag * dof; U_jik -= num_U_diag * dof;
        invD_jik -= dof*dof;

        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b_jik[f] / sqD_jik[f];

        for (idx_t d = 0; d < num_L_diag; d++) {
            xa[d] -= dof;
            sqDa[d] -= dof;
            vecvec_mla<idx_t, data_t, calc_t, dof>(L_jik + d*dof, sqDa[d], xa[d], tmp);
        }
        for (idx_t d = num_L_diag + 1; d < num_diag; d++) {
            xa[d] -= dof;
            sqDa[d] -= dof;
            const idx_t p = d - num_L_diag - 1;
            vecvec_mla<idx_t, data_t, calc_t, dof>(U_jik + p*dof, sqDa[d], xa[d], tmp);
        }
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, tmp2, - wgt);
        vecvec_div<idx_t, calc_t, dof>(tmp2, sqD_jik);
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            x_jik[f] = one_minus_weight * x_jik[f] + tmp2[f];
    }
}

#endif