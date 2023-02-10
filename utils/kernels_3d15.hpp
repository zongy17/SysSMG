#ifndef SYSSMG_KERNELS_3D15_HPP
#define SYSSMG_KERNELS_3D15_HPP

// ===================================================================
// ========================  3d15  kernels  ==========================
// ===================================================================
// #include "blk_kernels.hpp"
        /*  
                  /-------12------14                    
                 / |    8       10|                   
                /--|-----------/  |
               |   |           |  |
               |   |------11---|--13
   z   y       | 5 |    7      | 9|
   ^  ^        1---|-- 3 ------|  |
   | /         |   |           |  |
   |/          |   /-----------|--/ 
   O-------> x | 4      6      | / 
               0/------2-------|/
        */

// ====================== SPMV ===================================
/*
template<typename idx_t, typename data_t, typename calc_t, int dof>
void AOS_spmv_3d15(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size,
    const data_t * A_jik, const calc_t * x7, calc_t * y7, const calc_t * dummy)
{
    constexpr idx_t elms = dof*dof;
    const calc_t* x3 = x7 - vec_dki_size,
                * x11= x7 + vec_dki_size;
    for (idx_t k = 0; k < num; k++) {
        memset(y7, 0, sizeof(calc_t) * dof);
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik          , x3  - vec_dk_size - dof, y7);// 0
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +    elms, x3  - vec_dk_size      , y7);// 1
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 2* elms, x3                - dof, y7);// 2
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 3* elms, x3                     , y7);// 3

        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 4* elms, x7  - vec_dk_size - dof, y7);// 4
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 5* elms, x7  - vec_dk_size      , y7);// 5
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 6* elms, x7                - dof, y7);// 6
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 7* elms, x7                     , y7);// 6
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 8* elms, x7                + dof, y7);
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 9* elms, x7  + vec_dk_size      , y7);
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +10* elms, x7  + vec_dk_size + dof, y7);

        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +11* elms, x11                    , y7);
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +12* elms, x11               + dof, y7);
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +13* elms, x11 + vec_dk_size      , y7);
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +14* elms, x11 + vec_dk_size + dof, y7);

        A_jik += 15 * elms;
        x3 += dof; x7 += dof; x11 += dof;
        y7 += dof;
    }
}

template<int dof>
void AOS_spmv_3d15_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size,
    const __fp16 * A_jik, const float * x7, float * y7, const float * dummy)
{
    const float * xa[15];
    xa[7]  = x7;
    xa[3]  = x7  - vec_dki_size; xa[5]  = x7  - vec_dk_size;
    xa[11] = x7  + vec_dki_size; xa[9]  = x7  + vec_dk_size;
    xa[6]  = x7  - dof         ; xa[8]  = x7  + dof;
    xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
    xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;
    xa[0]  = xa[3] - vec_dk_size - dof;
    xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;
    xa[14] = xa[11]+ vec_dk_size + dof;
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
            for (int d = 0; d < 15; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 405, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);
                // 15 * 9 = 135
                A0_2_16 = vld3_f16(aos_ptr + 135); __builtin_prefetch(aos_ptr + 540, 0, 0);
                b0 = vcvt_f32_f16(A0_2_16.val[0]); b1 = vcvt_f32_f16(A0_2_16.val[1]); b2 = vcvt_f32_f16(A0_2_16.val[2]);
                s0 = vmlaq_n_f32(s0, b0, xp[0])  ; s1 = vmlaq_n_f32(s1, b1, xp[1])  ; s2 = vmlaq_n_f32(s2, b2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);

                A0_2_16 = vld3_f16(aos_ptr + 270); __builtin_prefetch(aos_ptr + 675, 0, 0);
                a0 = vcvt_f32_f16(A0_2_16.val[0]); a1 = vcvt_f32_f16(A0_2_16.val[1]); a2 = vcvt_f32_f16(A0_2_16.val[2]);
                t0 = vmlaq_n_f32(t0, a0, xp[0])  ; t1 = vmlaq_n_f32(t1, a1, xp[1])  ; t2 = vmlaq_n_f32(t2, a2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);

                aos_ptr += elms;
            }
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            s0 = vaddq_f32(s0, s1); s0 = vaddq_f32(s0, s2);
            t0 = vaddq_f32(t0, t1); t0 = vaddq_f32(t0, t2);
            vst1q_f32(y7, r0); y7 += dof; __builtin_prefetch(y7 + dof,1);
            vst1q_f32(y7, s0); y7 += dof; __builtin_prefetch(y7 + dof,1);
            vst1q_f32(y7, t0); y7 += dof; __builtin_prefetch(y7 + dof,1);
            
            A_jik += 45 * elms;// 15 * 3 = 45
        }
        for (k = 0; k < num - max_3k; k++) {// 做完剩下的元素
            const __fp16 * aos_ptr = A_jik; 
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            for (int d = 0; d < 15; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 135, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ; 
                xp += dof; __builtin_prefetch(xp + dof, 0);

                aos_ptr += elms;
            }
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            vst1q_f32(y7, r0); y7 += dof; __builtin_prefetch(y7 + dof,1);
            A_jik += 15 * elms;
        }
    }
}
*/

// ====================== PGS ===================================
/*
template<typename idx_t, typename data_t, typename calc_t, int dof>
void AOS_point_forward_zero_3d15(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * dummy0, const data_t * invD_jik,
    const calc_t * b7, calc_t * x7, const calc_t * dummy1)
{
    constexpr idx_t elms = dof*dof;
    const calc_t* x3 = x7 - vec_dki_size, * x5 = x7 - vec_dk_size ,
                * x1 = x3 - vec_dk_size;
    calc_t tmp[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b7[f];
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik         , x1 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik +   elms, x1      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 2*elms, x3 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 3*elms, x3      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 4*elms, x5 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 5*elms, x5      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 6*elms, x7 - dof, tmp);

        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, x7, - wgt);
        
        // printf(" k %d res %.6e %.6e %.6e\n", k, x7[0], x7[1], x7[2]);

        x1 += dof; x3 += dof; x5 += dof;
        x7 += dof; b7 += dof;
        L_jik += 7 * elms;
        invD_jik += elms;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void AOS_point_forward_ALL_3d15(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * invD_jik,
    const calc_t * b7, calc_t * x7, const calc_t * dummy1)
{
    constexpr idx_t elms = dof*dof;
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t* x3 = x7 - vec_dki_size, * x11 = x7 + vec_dki_size,
                * x5 = x7 - vec_dk_size , * x9  = x7 + vec_dk_size ,
                * x1 = x3 - vec_dk_size , * x13 = x11+ vec_dk_size;
    calc_t tmp[dof], tmp2[dof];
    for (idx_t k = 0; k < num; k++) {
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b7[f];
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik         , x1 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik +   elms, x1      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 2*elms, x3 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 3*elms, x3      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 4*elms, x5 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 5*elms, x5      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 6*elms, x7 - dof, tmp);

        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik         , x7 + dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik +   elms, x9      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 2*elms, x9 + dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 3*elms, x11     , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 4*elms, x11+ dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 5*elms, x13     , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 6*elms, x13+ dof, tmp);

        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, tmp2, - wgt);
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            x7[f] = one_minus_weight * x7[f] + tmp2[f];

        x1 += dof; x3 += dof; x5 += dof;
        x7 += dof; b7 += dof;
        x9 += dof; x11+= dof; x13+= dof;
        L_jik += 7 * elms; U_jik += 7 * elms;
        invD_jik += elms;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void AOS_point_backward_ALL_3d15(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size, const calc_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * invD_jik,
    const calc_t * b7, calc_t * x7, const calc_t * dummy1)
{
    constexpr idx_t elms = dof*dof;
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t* x3 = x7 - vec_dki_size, * x11 = x7 + vec_dki_size,
                * x5 = x7 - vec_dk_size , * x9  = x7 + vec_dk_size ,
                * x1 = x3 - vec_dk_size , * x13 = x11+ vec_dk_size;
    calc_t tmp[dof], tmp2[dof];
    for (idx_t k = 0; k < num; k++) {
        x1 -= dof; x3 -= dof; x5 -= dof;
        x7 -= dof; b7 -= dof;
        x9 -= dof; x11-= dof; x13-= dof;
        L_jik -= 7 * elms; U_jik -= 7 * elms;
        invD_jik -= elms;

        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            tmp[f] = - b7[f];
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik         , x1 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik +   elms, x1      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 2*elms, x3 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 3*elms, x3      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 4*elms, x5 - dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 5*elms, x5      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(L_jik + 6*elms, x7 - dof, tmp);

        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik         , x7 + dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik +   elms, x9      , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 2*elms, x9 + dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 3*elms, x11     , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 4*elms, x11+ dof, tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 5*elms, x13     , tmp);
        matvec_mla<idx_t, data_t, calc_t, dof>(U_jik + 6*elms, x13+ dof, tmp);

        matvec_mul<idx_t, data_t, calc_t, dof>(invD_jik, tmp, tmp2, - wgt);
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            x7[f] = one_minus_weight * x7[f] + tmp2[f];
    }
}
*/
/*
template<int dof>
void AOS_point_forward_ALL_3d15_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * U_jik, const __fp16 * invD_jik,
    const float * b7, float * x7, const float * dummy)
{
    constexpr int elms = dof*dof;
    const float * xa[15];
    xa[7]  = x7;
    xa[3]  = x7  - vec_dki_size; xa[5]  = x7  - vec_dk_size;
    xa[11] = x7  + vec_dki_size; xa[9]  = x7  + vec_dk_size;
    xa[6]  = x7  - dof         ; xa[8]  = x7  + dof;
    xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
    xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;
    xa[0]  = xa[3] - vec_dk_size - dof;
    xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;
    xa[14] = xa[11]+ vec_dk_size + dof;
    const float32x4_t vwgts = vdupq_n_f32(wgt), vone_minus_wgts = vdupq_n_f32(1.0 - wgt);
    if constexpr (dof == 3) {
        const __fp16 * aos_ptr = nullptr;
        float16x4x3_t A0_2_16;
        float32x4_t c0, c1, c2, tmp0;

        for (int k = 0; k < num; k++) {
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            tmp0 = vld1q_f32(b7); b7 += dof; __builtin_prefetch(b7 + 2*dof, 0, 0);

            aos_ptr = L_jik;// L部分
            for (int d = 0; d < 7; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 126, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ; 
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            L_jik += 7 * elms;

            aos_ptr = U_jik;// U部分
            for (int d = 8; d < 15; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 126, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ; 
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            U_jik += 7 * elms;

            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            tmp0 = vsubq_f32(tmp0, r0);// 此时tmp暂存了 b - 非本位的aj*xj
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            A0_2_16 = vld3_f16(invD_jik); invD_jik += elms; __builtin_prefetch(invD_jik + 18, 0, 0);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            tmp0 = vld1q_f32(x7);// 此时tmp0暂存原来的x解
            r0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w*D^{-1}*(b - 非本位的aj*xj)
            float buf[4]; vst1q_f32(buf, r0);
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x7[f] = buf[f];
            
            // printf(" k %d res %.6e %.6e %.6e\n", k, x7[0], x7[1], x7[2]);
            x7 += dof; __builtin_prefetch(x7 + 2*dof,1);
        }
    }
}

template<int dof>
void AOS_point_backward_ALL_3d15_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * U_jik, const __fp16 * invD_jik,
    const float * b7, float * x7, const float * dummy)
{
    constexpr int elms = dof*dof;
    const float * xa[15];
    xa[7]  = x7;
    xa[3]  = x7  - vec_dki_size; xa[5]  = x7  - vec_dk_size;
    xa[11] = x7  + vec_dki_size; xa[9]  = x7  + vec_dk_size;
    xa[6]  = x7  - dof         ; xa[8]  = x7  + dof;
    xa[2]  = xa[3] - dof       ; xa[1]  = xa[3] - vec_dk_size;
    xa[4]  = xa[5] - dof       ; xa[10] = xa[9] + dof;
    xa[0]  = xa[3] - vec_dk_size - dof;
    xa[12] = xa[11]+ dof       ; xa[13] = xa[11]+ vec_dk_size;
    xa[14] = xa[11]+ vec_dk_size + dof;
    const float32x4_t vwgts = vdupq_n_f32(wgt), vone_minus_wgts = vdupq_n_f32(1.0 - wgt);
    if constexpr (dof == 3) {
        const __fp16 * aos_ptr = nullptr;
        float16x4x3_t A0_2_16;
        float32x4_t c0, c1, c2, tmp0;

        for (int k = 0; k < num; k++) {// 做完剩下的元素
            b7 -= dof; __builtin_prefetch(b7 - 2*dof, 0, 0);
            float32x4_t r0 = vld1q_f32(b7), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存

            L_jik -= 7 * elms; aos_ptr = L_jik;
            for (int d = 0; d < 7; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr - 126, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                xp -= dof; __builtin_prefetch(xp - 2*dof, 0);
                r0 = vmlsq_n_f32(r0, c0, xp[0])  ; r1 = vmlsq_n_f32(r1, c1, xp[1])  ; r2 = vmlsq_n_f32(r2, c2, xp[2])  ; 
                aos_ptr += elms;
            }

            U_jik -= 7 * elms; aos_ptr = U_jik;
            for (int d = 8; d < 15; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr - 126, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                xp -= dof; __builtin_prefetch(xp - 2*dof, 0);
                r0 = vmlsq_n_f32(r0, c0, xp[0])  ; r1 = vmlsq_n_f32(r1, c1, xp[1])  ; r2 = vmlsq_n_f32(r2, c2, xp[2])  ; 
                aos_ptr += elms;
            }

            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            tmp0 = vmulq_f32(r0, vwgts);// 此时tmp暂存了 w*(b - 非本位的aj*xj)

            invD_jik -= elms;
            A0_2_16 = vld3_f16(invD_jik); __builtin_prefetch(invD_jik - 18, 0, 0);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);

            x7 -= dof; __builtin_prefetch(x7 - 2*dof,1);
            tmp0 = vld1q_f32(x7);// 此时tmp0暂存原来的x解
            tmp0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w*D^{-1}*(b - 非本位的aj*xj)

            float buf[4]; vst1q_f32(buf, tmp0);
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x7[f] = buf[f];
        }
    }
}
*/

#endif