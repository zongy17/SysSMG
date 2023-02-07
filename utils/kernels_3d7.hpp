#ifndef SYSSMG_KERNELS_3D7_HPP
#define SYSSMG_KERNELS_3D7_HPP

// ===================================================================
// ========================  3d7  kernels  ==========================
// ===================================================================
#include "blk_kernels.hpp"
        /*  
                  /--------------/                    
                 / |    4       / |                   
                /--|-----------/  |
               |   |           |  |
               |   |------6----|--|
   z   y       | 1 |    3      | 5|
   ^  ^        |---|-- 0 ------|  |
   | /         |   |           |  |
   |/          |   /-----------|--/ 
   O-------> x | /      2      | / 
               |/--------------|/

        */

// ====================== SPMV ===================================
/*
template<typename idx_t, typename data_t, typename calc_t, int dof>
void AOS_spmv_3d7(const idx_t num,
    const idx_t vec_dk_size, const idx_t vec_dki_size,
    const data_t * A_jik, const calc_t * x3, calc_t * y3, const calc_t * dummy)
{
    constexpr idx_t elms = dof*dof;
    const calc_t* x0 = x3 - vec_dki_size,
                * x6 = x3 + vec_dki_size;
    for (idx_t k = 0; k < num; k++) {
        memset(y3, 0, sizeof(calc_t) * dof);
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik          , x0               , y3);// 0
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik +    elms, x3  - vec_dk_size, y3);// 1
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 2* elms, x3  - dof        , y3);// 2
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 3* elms, x3               , y3);// 3
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 4* elms, x3  + dof        , y3);// 4
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 5* elms, x3  + vec_dk_size, y3);// 5
        matvec_mla<idx_t, data_t, calc_t, dof>(A_jik + 6* elms, x6               , y3);// 6

        A_jik += 7 * elms;
        x0 += dof; x3 += dof; x6 += dof;
        y3 += dof;
    }
}


template<int dof>
void AOS_spmv_3d7_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size,
    const __fp16 * A_jik, const float * x3, float * y3, const float * dummy)
{
    const float * xa[7];
    xa[3] = x3;
    xa[0] = x3 - vec_dki_size; xa[6] = x3 + vec_dki_size;
    xa[1] = x3 - vec_dk_size ; xa[5] = x3 + vec_dk_size ;
    xa[2] = x3 - dof         ; xa[4] = x3 + dof         ;
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
            for (int d = 0; d < 7; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 189, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);
                // 7 * 9 = 63
                A0_2_16 = vld3_f16(aos_ptr +  63); __builtin_prefetch(aos_ptr + 252, 0, 0);
                b0 = vcvt_f32_f16(A0_2_16.val[0]); b1 = vcvt_f32_f16(A0_2_16.val[1]); b2 = vcvt_f32_f16(A0_2_16.val[2]);
                s0 = vmlaq_n_f32(s0, b0, xp[0])  ; s1 = vmlaq_n_f32(s1, b1, xp[1])  ; s2 = vmlaq_n_f32(s2, b2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);

                A0_2_16 = vld3_f16(aos_ptr + 126); __builtin_prefetch(aos_ptr + 315, 0, 0);
                a0 = vcvt_f32_f16(A0_2_16.val[0]); a1 = vcvt_f32_f16(A0_2_16.val[1]); a2 = vcvt_f32_f16(A0_2_16.val[2]);
                t0 = vmlaq_n_f32(t0, a0, xp[0])  ; t1 = vmlaq_n_f32(t1, a1, xp[1])  ; t2 = vmlaq_n_f32(t2, a2, xp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);

                aos_ptr += elms;
            }
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            s0 = vaddq_f32(s0, s1); s0 = vaddq_f32(s0, s2);
            t0 = vaddq_f32(t0, t1); t0 = vaddq_f32(t0, t2);
            vst1q_f32(y3, r0); y3 += dof; __builtin_prefetch(y3 + dof,1);
            vst1q_f32(y3, s0); y3 += dof; __builtin_prefetch(y3 + dof,1);
            vst1q_f32(y3, t0); y3 += dof; __builtin_prefetch(y3 + dof,1);
            
            A_jik += 21 * elms;// 7 * 3 = 21
        }
        for (k = 0; k < num - max_3k; k++) {// 做完剩下的元素
            const __fp16 * aos_ptr = A_jik; 
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            for (int d = 0; d < 7; d++) {
                const float * & xp = xa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr +  63, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2])  ; 
                xp += dof; __builtin_prefetch(xp + dof, 0);

                aos_ptr += elms;
            }
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            vst1q_f32(y3, r0); y3 += dof; __builtin_prefetch(y3 + dof,1);
            A_jik += 7 * elms;
        }
    }
}
*/

#endif