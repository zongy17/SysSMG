#ifndef SYSSMG_KERNELS_3D27_HPP
#define SYSSMG_KERNELS_3D27_HPP

// ===================================================================
// ========================  3d27  kernels  ==========================
// ===================================================================
/*  
                  /20--- 23------26
                 11|    14      17|
               2---|---5-------8  |
               |   |           |  |
               |   19----22----|-25
   z   y       | 10|    13     |16|
   ^  ^        1---|-- 4 ------7  |
   | /         |   |           |  |
   |/          |   18----21----|-24 
   O-------> x | 9      12     |15 
               0/------3-------6/

        */
// ====================== SPMV ===================================
/*
template<typename idx_t, typename data_t, typename calc_t, int dof>
void AOS_spmv_3d27(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size,
    const data_t * A_jik, const calc_t * x13, calc_t * y13, const calc_t * dummy)
{
    constexpr idx_t elms = dof*dof;
    static_assert(sizeof(data_t) == sizeof(calc_t));
    const calc_t* x4 = x13 - vec_dki_size,
                * x22= x13 + vec_dki_size;
    for (idx_t k = 0; k < num; k++) {
        memset(y13, 0, sizeof(calc_t) * dof);
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik          , x4  - vec_dk_size - dof, y13);// 0
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +    elms, x4  - vec_dk_size      , y13);// 1
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 2* elms, x4  - vec_dk_size + dof, y13);// 2
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 3* elms, x4                - dof, y13);// 3
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 4* elms, x4                     , y13);// 4
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 5* elms, x4                + dof, y13);// 5
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 6* elms, x4  + vec_dk_size - dof, y13);// 6
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 7* elms, x4  + vec_dk_size      , y13);// 7
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 8* elms, x4  + vec_dk_size + dof, y13);

        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 9* elms, x13 - vec_dk_size - dof, y13);// 0
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +10* elms, x13 - vec_dk_size      , y13);// 1
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +11* elms, x13 - vec_dk_size + dof, y13);// 2
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +12* elms, x13               - dof, y13);// 3
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +13* elms, x13                    , y13);// 4
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +14* elms, x13               + dof, y13);// 5
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +15* elms, x13 + vec_dk_size - dof, y13);// 6
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +16* elms, x13 + vec_dk_size      , y13);// 7
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +17* elms, x13 + vec_dk_size + dof, y13);

        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +18* elms, x22 - vec_dk_size - dof, y13);// 0
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +19* elms, x22 - vec_dk_size      , y13);// 1
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +20* elms, x22 - vec_dk_size + dof, y13);// 2
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +21* elms, x22               - dof, y13);// 3
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +22* elms, x22                    , y13);// 4
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +23* elms, x22               + dof, y13);// 5
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +24* elms, x22 + vec_dk_size - dof, y13);// 6
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +25* elms, x22 + vec_dk_size      , y13);// 7
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +26* elms, x22 + vec_dk_size + dof, y13);

        A_jik += 27 * elms;
        x4  += dof; x13 += dof; x22 += dof;
        y13 += dof;
    }
}
*/

/*
template<int dof>
void AOS_spmv_3d27_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size,
    const __fp16 * A_jik, const float * x13, float * y13, const float * dummy)
{
    const float * xa[27];
    xa[13] = x13;
    xa[ 4] = x13  - vec_dki_size; xa[22] = x13  + vec_dki_size;
    xa[10] = x13  - vec_dk_size ; xa[16] = x13  + vec_dk_size ;
    xa[12] = x13  - dof         ; xa[14] = x13  + dof         ;
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
    constexpr int elms = dof*dof;
    if constexpr (dof == 3) {
        float16x4x3_t   A0_2_16;
        float32x4_t c0, c1, c2;
        int k = 0, max_3k = num & (~2);
        for ( ; k < max_3k; k += 3) {
            const __fp16 * aos_ptr = A_jik;
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0),// 第几列的结果暂存
                        s0 = vdupq_n_f32(0.0), s1 = vdupq_n_f32(0.0), s2 = vdupq_n_f32(0.0),
                        t0 = vdupq_n_f32(0.0), t1 = vdupq_n_f32(0.0), t2 = vdupq_n_f32(0.0);
            float32x4_t b0, b1, b2, a0, a1, a2;
            for (int d = 0; d < 27; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 729, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);
                // 27 * 9 = 243
                A0_2_16 = vld3_f16(aos_ptr + 243); __builtin_prefetch(aos_ptr + 972, 0, 0);
                b0 = vcvt_f32_f16(A0_2_16.val[0]); b1 = vcvt_f32_f16(A0_2_16.val[1]); b2 = vcvt_f32_f16(A0_2_16.val[2]);
                s0 = vmlaq_n_f32(s0, b0, xp[0])  ; s1 = vmlaq_n_f32(s1, b1, xp[1])  ; s2 = vmlaq_n_f32(s2, b2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);

                A0_2_16 = vld3_f16(aos_ptr + 486); __builtin_prefetch(aos_ptr + 1215, 0, 0);
                a0 = vcvt_f32_f16(A0_2_16.val[0]); a1 = vcvt_f32_f16(A0_2_16.val[1]); a2 = vcvt_f32_f16(A0_2_16.val[2]);
                t0 = vmlaq_n_f32(t0, a0, xp[0])  ; t1 = vmlaq_n_f32(t1, a1, xp[1])  ; t2 = vmlaq_n_f32(t2, a2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);

                aos_ptr += elms;
            }
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            s0 = vaddq_f32(s0, s1); s0 = vaddq_f32(s0, s2);
            t0 = vaddq_f32(t0, t1); t0 = vaddq_f32(t0, t2);
            vst1q_f32(y13, r0); y13 += dof; __builtin_prefetch(y13 + dof,1);
            vst1q_f32(y13, s0); y13 += dof; __builtin_prefetch(y13 + dof,1);
            vst1q_f32(y13, t0); y13 += dof; __builtin_prefetch(y13 + dof,1);
            
            A_jik += 81 * elms;// 27 * 3 = 81
        }
        for (k = 0; k < num - max_3k; k++) {// 做完剩下的元素
            const __fp16 * aos_ptr = A_jik; 
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            for (int d = 0; d < 27; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 243, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ; 
                xp += dof; __builtin_prefetch(xp + dof, 0);

                aos_ptr += elms; 
            }
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            vst1q_f32(y13, r0); y13 += dof; __builtin_prefetch(y13 + dof,1);
            A_jik += 27 * elms;
        }
    }
}
*/

// ========================== PGS ===============================
/*
template<typename idx_t, typename data_t, typename calc_t, int dof>
void AOS_point_forward_zero_3d27(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * dummy0, const data_t * invD_jik,
    const calc_t * b13, calc_t * x13, const calc_t * dummy1)
{
    constexpr idx_t elms = dof*dof;
    const calc_t* x4 = x13 - vec_dki_size, * x10 = x13 - vec_dk_size,
                * x1 = x13 - vec_dki_size - vec_dk_size,
                * x7 = x13 - vec_dki_size + vec_dk_size;
    calc_t tmp[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b13[f];
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik         , x1 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik +   elms, x1      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 2*elms, x1 + dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 3*elms, x4 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 4*elms, x4      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 5*elms, x4 + dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 6*elms, x7 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 7*elms, x7      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 8*elms, x7 + dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 9*elms, x10- dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik +10*elms, x10     , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik +11*elms, x10+ dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik +12*elms, x13- dof, tmp);

        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, x13, - wgt);

        x1  += dof; x4  += dof; x7 += dof; x10 += dof;
        x13 += dof; b13 += dof;
        L_jik += 13 * elms;
        invD_jik += elms;
    }
}
*/

// =========================================================================
// =========================== Structure Of Array ==========================
// =========================================================================

#endif