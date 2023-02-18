#ifndef SYSSMG_VECT_KERNELS_HPP
#define SYSSMG_VECT_KERNELS_HPP

// ==========================================================
//  ==================== Vectorized Kernels ====================
// ==========================================================
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include "scal_kernels.hpp"
// =================================== SPMV ================================



template<int dof, int num_diag>
void AOS_spmv_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size,
    const __fp16 * A_jik, const float * x_jik, float * y_jik, const float * dummy)
{
    const float * xa[num_diag];
    spmv_prepare_xa;
    constexpr int elms = dof*dof;
    if constexpr (dof == 3) {
        float16x4x3_t   A0_2_16;
        float32x4_t c0, c1, c2;
        int k = 0, max_3k = (num / 3) * 3;
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

template<int dof, int num_diag>
void AOS_spmv_3d_scaled_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size,
    const __fp16 * A_jik, const float * x_jik, float * y_jik, const float * sqD_jik)
{
    const float * xa[num_diag], * sqDa[num_diag];
    spmv_prepare_xa;
    for (int d = 0; d < num_diag; d++)
        sqDa[d] = sqD_jik + (xa[d] - x_jik);
    constexpr int elms = dof*dof;
    if constexpr (dof == 3) {
        float16x4x3_t   A0_2_16;
        float32x4_t c0, c1, c2;
        int k = 0, max_3k = (num / 3) * 3;
        constexpr int mat_prft = elms * num_diag;
        for ( ; k < max_3k; k += 3) {
            const __fp16 * aos_ptr = A_jik;
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0),// 第几列的结果暂存
                        s0 = vdupq_n_f32(0.0), s1 = vdupq_n_f32(0.0), s2 = vdupq_n_f32(0.0),
                        t0 = vdupq_n_f32(0.0), t1 = vdupq_n_f32(0.0), t2 = vdupq_n_f32(0.0);
            float32x4_t b0, b1, b2, a0, a1, a2;
            for (int d = 0; d < num_diag; d++) {
                const float * & xp = xa[d], * & sqDp = sqDa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 3*mat_prft, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0]*sqDp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1]*sqDp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2]*sqDp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);
                sqDp += dof; __builtin_prefetch(sqDp + dof*3, 0);
                // 15 * 9 = 135
                A0_2_16 = vld3_f16(aos_ptr + mat_prft); __builtin_prefetch(aos_ptr + 4*mat_prft, 0, 0);
                b0 = vcvt_f32_f16(A0_2_16.val[0]); b1 = vcvt_f32_f16(A0_2_16.val[1]); b2 = vcvt_f32_f16(A0_2_16.val[2]);
                s0 = vmlaq_n_f32(s0, b0, xp[0]*sqDp[0])  ; s1 = vmlaq_n_f32(s1, b1, xp[1]*sqDp[1])  ; s2 = vmlaq_n_f32(s2, b2, xp[2]*sqDp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);
                sqDp += dof; __builtin_prefetch(sqDp + dof*3, 0);

                A0_2_16 = vld3_f16(aos_ptr + 2*mat_prft); __builtin_prefetch(aos_ptr + 5*mat_prft, 0, 0);
                a0 = vcvt_f32_f16(A0_2_16.val[0]); a1 = vcvt_f32_f16(A0_2_16.val[1]); a2 = vcvt_f32_f16(A0_2_16.val[2]);
                t0 = vmlaq_n_f32(t0, a0, xp[0]*sqDp[0])  ; t1 = vmlaq_n_f32(t1, a1, xp[1]*sqDp[1])  ; t2 = vmlaq_n_f32(t2, a2, xp[2]*sqDp[2])  ;
                xp += dof; __builtin_prefetch(xp + dof*3, 0);
                sqDp += dof; __builtin_prefetch(sqDp + dof*3, 0);

                aos_ptr += elms;
            }
            
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2); r1 = vld1q_f32(sqD_jik); r0 = vmulq_f32(r0, r1); sqD_jik += dof;
            s0 = vaddq_f32(s0, s1); s0 = vaddq_f32(s0, s2); s1 = vld1q_f32(sqD_jik); s0 = vmulq_f32(s0, s1); sqD_jik += dof;
            t0 = vaddq_f32(t0, t1); t0 = vaddq_f32(t0, t2); t1 = vld1q_f32(sqD_jik); t0 = vmulq_f32(t0, t1); sqD_jik += dof;
            vst1q_f32(y_jik, r0); y_jik += dof; __builtin_prefetch(y_jik + dof,1);
            vst1q_f32(y_jik, s0); y_jik += dof; __builtin_prefetch(y_jik + dof,1);
            vst1q_f32(y_jik, t0); y_jik += dof; __builtin_prefetch(y_jik + dof,1);
            
            A_jik += 3*num_diag * elms;// 15 * 3 = 45
        }
        for (k = 0; k < num - max_3k; k++) {// 做完剩下的元素
            const __fp16 * aos_ptr = A_jik; 
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            for (int d = 0; d < num_diag; d++) {
                const float * & xp = xa[d], * & sqDp = sqDa[d];// 取引用！
                A0_2_16 = vld3_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + mat_prft, 0, 0);
                c0 = vcvt_f32_f16(A0_2_16.val[0]); c1 = vcvt_f32_f16(A0_2_16.val[1]); c2 = vcvt_f32_f16(A0_2_16.val[2]);
                r0 = vmlaq_n_f32(r0, c0, xp[0]*sqDp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1]*sqDp[1])  ; r2 = vmlaq_n_f32(r2, c2, xp[2]*sqDp[2])  ; 
                xp += dof; __builtin_prefetch(xp + dof, 0);
                sqDp += dof; __builtin_prefetch(sqDp + dof, 0);

                aos_ptr += elms;
            }
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2); r1 = vld1q_f32(sqD_jik); r0 = vmulq_f32(r0, r1); sqD_jik += dof;
            vst1q_f32(y_jik, r0); y_jik += dof; __builtin_prefetch(y_jik + dof,1);
            A_jik += num_diag * elms;
        }
    }
}

/*
template<int dof, int num_L_diag, int num_U_diag>
void AOS_compress_spmv_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size,
    const __fp16 * L_jik, const __fp16 * D_jik, const __fp16 * U_jik,
    const float * x_jik, float * y_jik, const float * dummy)
{
    static_assert(num_L_diag == 3 && num_U_diag == 3);
    constexpr int num_diag = num_L_diag + 1 + num_U_diag;
    static_assert(num_diag >> 1 == num_L_diag && num_L_diag == num_U_diag);
    const float *x0 = x_jik - vec_dki_size, * x6 = x_jik + vec_dki_size,
                *x1 = x_jik - vec_dk_size , * x5 = x_jik + vec_dk_size ,
                *x2 = x_jik - dof         , * x4 = x_jik + dof         ;
    if constexpr (dof == 3) {
        float16x4x3_t Diag_16;
        float16x4_t A0_16, A1_16, A2_16;
        float32x4_t A0_32, A1_32, A2_32, x0_32, x1_32, x2_32;
        float32x4_t c0, c1, c2;
        constexpr int mat_prft = dof * num_L_diag;
        for (int k = 0; k < num; k++) {
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            
            A0_16 = vld1_f16(L_jik      ); x0_32 = vld1q_f32(x0); x0 += dof; __builtin_prefetch(x0 + 3*dof, 0);
            A0_32 = vcvt_f32_f16(A0_16); r0   = vmlaq_f32(r0, A0_32, x0_32);
            A1_16 = vld1_f16(L_jik+  dof); x1_32 = vld1q_f32(x1); x1 += dof; __builtin_prefetch(x1 + 3*dof, 0);
            A1_32 = vcvt_f32_f16(A1_16); r1   = vmlaq_f32(r1, A1_32, x1_32);
            A2_16 = vld1_f16(L_jik+2*dof); x2_32 = vld1q_f32(x2); x2 += dof; __builtin_prefetch(x2 + 3*dof, 0);
            A2_32 = vcvt_f32_f16(A2_16); r2   = vmlaq_f32(r2, A2_32, x2_32);
            L_jik += num_L_diag * dof; __builtin_prefetch(L_jik + 3*mat_prft, 0, 0);

            A0_16 = vld1_f16(U_jik      ); x0_32 = vld1q_f32(x4); x4 += dof; __builtin_prefetch(x4 + 3*dof, 0);
            A0_32 = vcvt_f32_f16(A0_16); r0   = vmlaq_f32(r0, A0_32, x0_32);
            A1_16 = vld1_f16(U_jik+  dof); x1_32 = vld1q_f32(x5); x5 += dof; __builtin_prefetch(x5 + 3*dof, 0);
            A1_32 = vcvt_f32_f16(A1_16); r1   = vmlaq_f32(r1, A1_32, x1_32);
            A2_16 = vld1_f16(U_jik+2*dof); x2_32 = vld1q_f32(x6); x6 += dof; __builtin_prefetch(x6 + 3*dof, 0);
            A2_32 = vcvt_f32_f16(A2_16); r2   = vmlaq_f32(r2, A2_32, x2_32);
            U_jik += num_U_diag * dof; __builtin_prefetch(U_jik + 3*mat_prft, 0, 0);

            Diag_16 = vld3_f16(D_jik); D_jik += dof*dof; __builtin_prefetch(D_jik + 3*dof*dof, 0, 0);
            c0 = vcvt_f32_f16(Diag_16.val[0]) ; c1 = vcvt_f32_f16(Diag_16.val[1]) ; c2 = vcvt_f32_f16(Diag_16.val[2]) ;
            r0 = vmlaq_n_f32(r0, c0, x_jik[0]); r1 = vmlaq_n_f32(r1, c1, x_jik[1]); r2 = vmlaq_n_f32(r2, c2, x_jik[2]);
            x_jik += dof; __builtin_prefetch(x_jik + 3*dof, 0);

            r0 = vaddq_f32(r0, r1);
            r0 = vaddq_f32(r0, r2);
            vst1q_f32(y_jik, r0); y_jik += dof; __builtin_prefetch(y_jik + 3*dof,1);
        }
    }
}
*/

template<int dof, int num_L_diag, int num_U_diag>
void AOS_compress_spmv_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size,
    const __fp16 * L_jik, const __fp16 * D_jik, const __fp16 * U_jik,
    const float * x_jik, float * y_jik, const float * dummy)
{
    constexpr int num_diag = num_L_diag + 1 + num_U_diag;
    static_assert(num_diag >> 1 == num_L_diag && num_L_diag == num_U_diag);
    const float * xa[num_diag];
    spmv_prepare_xa;
    constexpr int diag_id = num_L_diag;
    constexpr int mat_prft = dof * num_L_diag;
    if constexpr (dof == 3) {
        float16x4x3_t Diag_16;
        float32x4_t c0, c1, c2;
        float16x4_t AL_16, AU_16;
        float32x4_t AL_32, AU_32, xL_32, xU_32;
        for (int k = 0; k < num; k++) {
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            const __fp16 * L_ptr = L_jik, * U_ptr = U_jik;
            for (int d = 0; d < diag_id; d++) {// L 和 U
                const float * & xL = xa[d]      , * & xU = xa[d + num_L_diag + 1];
                AL_16 = vld1_f16(L_ptr); __builtin_prefetch(L_ptr + 3*mat_prft, 0, 0);
                AU_16 = vld1_f16(U_ptr); __builtin_prefetch(U_ptr + 3*mat_prft, 0, 0); 
                AL_32 = vcvt_f32_f16(AL_16)     ; AU_32 = vcvt_f32_f16(AU_16);
                xL_32 = vld1q_f32(xL)           ; xU_32 = vld1q_f32(xU);
                r0 = vmlaq_f32(r0, AL_32, xL_32); r1 = vmlaq_f32(r1, AU_32, xU_32);
                xL += dof; __builtin_prefetch(xL + 3*dof, 0);
                xU += dof; __builtin_prefetch(xU + 3*dof, 0);
                
                L_ptr += dof;
                U_ptr += dof;
            }
            L_jik += num_L_diag * dof;
            U_jik += num_U_diag * dof;

            Diag_16 = vld3_f16(D_jik); D_jik += dof*dof; __builtin_prefetch(D_jik + 3*dof*dof, 0, 0);
            c0 = vcvt_f32_f16(Diag_16.val[0]) ; c1 = vcvt_f32_f16(Diag_16.val[1]) ; c2 = vcvt_f32_f16(Diag_16.val[2]) ;
            r0 = vmlaq_n_f32(r0, c0, x_jik[0]); r1 = vmlaq_n_f32(r1, c1, x_jik[1]); r2 = vmlaq_n_f32(r2, c2, x_jik[2]);
            x_jik += dof; __builtin_prefetch(x_jik + 3*dof, 0);

            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            vst1q_f32(y_jik, r0); y_jik += dof; __builtin_prefetch(y_jik + 3*dof,1);
        }
    }
}

template<int dof, int num_L_diag, int num_U_diag>
void AOS_compress_spmv_3d_scaled_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size,
    const __fp16 * L_jik, const __fp16 * D_jik, const __fp16 * U_jik,
    const float * x_jik, float * y_jik, const float * sqD_jik)
{
    constexpr int num_diag = num_L_diag + 1 + num_U_diag;
    static_assert(num_diag >> 1 == num_L_diag && num_L_diag == num_U_diag);
    const float * xa[num_diag], * sqDa[num_diag];
    spmv_prepare_xa;
    for (int d = 0; d < num_diag; d++)
        sqDa[d] = sqD_jik + (xa[d] - x_jik);// 两者向量的偏移是一样的

    constexpr int diag_id = num_L_diag;
    constexpr int mat_prft = dof * num_L_diag;
    if constexpr (dof == 3) {
        float16x4x3_t Diag_16;
        float32x4_t c0, c1, c2;
        float16x4_t AL_16, AU_16;
        float32x4_t AL_32, AU_32, xL_32, xU_32, qL_32, qU_32;
        for (int k = 0; k < num; k++) {
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0), r2 = vdupq_n_f32(0.0);// 第几列的结果暂存
            // float32x4_t tL = vdupq_n_f32(0.0), tU = vdupq_n_f32(0.0);
            const __fp16 * L_ptr = L_jik, * U_ptr = U_jik;
            for (int d = 0; d < diag_id; d++) {// L 和 U
                const float * & xL = xa[d]      , * & xU = xa[d + num_L_diag + 1];
                const float * & qL = sqDa[d]    , * & qU = sqDa[d + num_L_diag + 1];
                AL_16 = vld1_f16(L_ptr); __builtin_prefetch(L_ptr + 3*mat_prft, 0, 0);
                AU_16 = vld1_f16(U_ptr); __builtin_prefetch(U_ptr + 3*mat_prft, 0, 0); 
                AL_32 = vcvt_f32_f16(AL_16)     ; AU_32 = vcvt_f32_f16(AU_16);
                xL_32 = vld1q_f32(xL)           ; xU_32 = vld1q_f32(xU);
                qL_32 = vld1q_f32(qL)           ; qU_32 = vld1q_f32(qU);
                AL_32 = vmulq_f32(AL_32, qL_32) ; AU_32 = vmulq_f32(AU_32, qU_32);
                // tL = vmlaq_f32(tL, AL_32, xL_32); tU = vmlaq_f32(tU, AU_32, xU_32);
                r0 = vmlaq_f32(r0, AL_32, xL_32); r1 = vmlaq_f32(r1, AU_32, xU_32);
                xL += dof; __builtin_prefetch(xL + 3*dof, 0);
                xU += dof; __builtin_prefetch(xU + 3*dof, 0);
                qL += dof; __builtin_prefetch(qL + 3*dof, 0);
                qU += dof; __builtin_prefetch(qU + 3*dof, 0);
                
                L_ptr += dof;
                U_ptr += dof;
            }
            L_jik += num_L_diag * dof;
            U_jik += num_U_diag * dof;
            
            // tL = vaddq_f32(tL, tU);

            float32x4_t my_sqD = vld1q_f32(sqD_jik); 

            Diag_16 = vld3_f16(D_jik); D_jik += dof*dof; __builtin_prefetch(D_jik + 3*dof*dof, 0, 0);
            c0 = vcvt_f32_f16(Diag_16.val[0]) ; c1 = vcvt_f32_f16(Diag_16.val[1]) ; c2 = vcvt_f32_f16(Diag_16.val[2]) ;
            c0 = vmulq_n_f32(c0, sqD_jik[0])  ; c1 = vmulq_n_f32(c1, sqD_jik[1])  ; c2 = vmulq_n_f32(c2, sqD_jik[2]); 
            r0 = vmlaq_n_f32(r0, c0, x_jik[0]); r1 = vmlaq_n_f32(r1, c1, x_jik[1]); r2 = vmlaq_n_f32(r2, c2, x_jik[2]);

            sqD_jik += dof; __builtin_prefetch(sqD_jik + 3*dof, 0);
            x_jik += dof; __builtin_prefetch(x_jik + 3*dof, 0);

            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);
            r0 = vmulq_f32(r0, my_sqD);
            vst1q_f32(y_jik, r0); 

            y_jik += dof; __builtin_prefetch(y_jik + 3*dof,1);
        }
    }
}

// ================================= PGS ====================================
//  - - - - - - - - - - - - - - - 正常的函数
template<int dof, int num_L_diag>
void AOS_point_forward_zero_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * dummy0, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * dummy1)
{
    constexpr int elms = dof*dof;
    const float * xa[num_L_diag];
    pgsf_prepare_xa;
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
    pgsAll_prepare_xa;
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
    pgsAll_prepare_xa;
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

//  - - - - - - - - - - - - - - - 压缩的函数

template<int dof, int num_L_diag>
void AOS_compress_point_forward_zero_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * dummy0, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * dummy1)
{
    static_assert(num_L_diag == 3);
    const float * x0 = x_jik - vec_dki_size, 
                * x1 = x_jik - vec_dk_size ,
                * x2 = x_jik - dof;
    const float32x4_t vwgts = vdupq_n_f32(wgt);
    if constexpr (dof == 3) {
        float16x4x3_t A0_2_16;
        float16x4_t A0_16, A1_16, A2_16;
        float32x4_t A0_32, A1_32, A2_32, x0_32, x1_32, x2_32, tmp0;
        float32x4_t c0, c1, c2, r0, r1, r2;
        for (int k = 0; k < num; k++) {
            tmp0 = vld1q_f32(b_jik); b_jik += dof; __builtin_prefetch(b_jik + 3*dof, 0, 0);

            A0_16 = vld1_f16(L_jik        ); A0_32 = vcvt_f32_f16(A0_16);
            A1_16 = vld1_f16(L_jik +   dof); A1_32 = vcvt_f32_f16(A1_16);
            A2_16 = vld1_f16(L_jik + 2*dof); A2_32 = vcvt_f32_f16(A2_16);
            L_jik += 3*dof; __builtin_prefetch(L_jik + 9*dof, 0, 0);
            x0_32 = vld1q_f32(x0); x0 += dof; __builtin_prefetch(x0 + 3*dof, 0, 0);
            x1_32 = vld1q_f32(x1); x1 += dof; __builtin_prefetch(x1 + 3*dof, 0, 0);
            x2_32 = vld1q_f32(x2); x2 += dof; __builtin_prefetch(x2 + 3*dof, 0, 0);
            tmp0 = vmlsq_f32(tmp0, A0_32, x0_32);
            tmp0 = vmlsq_f32(tmp0, A1_32, x1_32);
            tmp0 = vmlsq_f32(tmp0, A2_32, x2_32);// 此时tmp暂存了 b - 非本位的aj*xj
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            A0_2_16 = vld3_f16(invD_jik); invD_jik += dof*dof; __builtin_prefetch(invD_jik + 3*dof*dof, 0, 0);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            float buf[4]; vst1q_f32(buf, r0);// 不能直接存，必须先倒一趟，以免向量寄存器内的lane3会污染原数据
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x_jik[f] = buf[f];

            x_jik += dof; __builtin_prefetch(x_jik + 3*dof,1);
        }
    }
}

template<int dof, int num_L_diag, int num_U_diag>
void AOS_compress_point_forward_ALL_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * U_jik, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * dummy)
{
    static_assert(num_L_diag == 3);
    const float * x0 = x_jik - vec_dki_size, * x6 = x_jik + vec_dki_size,
                * x1 = x_jik - vec_dk_size , * x5 = x_jik + vec_dk_size ,
                * x2 = x_jik - dof         , * x4 = x_jik + dof         ;
    const float32x4_t vwgts = vdupq_n_f32(wgt), vone_minus_wgts = vdupq_n_f32(1.0 - wgt);
    if constexpr (dof == 3) {
        float16x4x3_t A0_2_16;
        float16x4_t A0_16, A1_16, A2_16;
        float32x4_t A0_32, A1_32, A2_32, x0_32, x1_32, x2_32, tmp0;
        float32x4_t c0, c1, c2, r0, r1, r2;
        for (int k = 0; k < num; k++) {
            tmp0 = vld1q_f32(b_jik); b_jik += dof; __builtin_prefetch(b_jik + 3*dof, 0, 0);

            A0_16 = vld1_f16(L_jik        ); A0_32 = vcvt_f32_f16(A0_16);
            A1_16 = vld1_f16(L_jik +   dof); A1_32 = vcvt_f32_f16(A1_16);
            A2_16 = vld1_f16(L_jik + 2*dof); A2_32 = vcvt_f32_f16(A2_16);
            L_jik += 3*dof; __builtin_prefetch(L_jik + 9*dof, 0, 0);
            x0_32 = vld1q_f32(x0); x0 += dof; __builtin_prefetch(x0 + 3*dof, 0, 0);
            x1_32 = vld1q_f32(x1); x1 += dof; __builtin_prefetch(x1 + 3*dof, 0, 0);
            x2_32 = vld1q_f32(x2); x2 += dof; __builtin_prefetch(x2 + 3*dof, 0, 0);
            tmp0 = vmlsq_f32(tmp0, A0_32, x0_32);
            tmp0 = vmlsq_f32(tmp0, A1_32, x1_32);
            tmp0 = vmlsq_f32(tmp0, A2_32, x2_32);
            
            A0_16 = vld1_f16(U_jik        ); A0_32 = vcvt_f32_f16(A0_16);
            A1_16 = vld1_f16(U_jik +   dof); A1_32 = vcvt_f32_f16(A1_16);
            A2_16 = vld1_f16(U_jik + 2*dof); A2_32 = vcvt_f32_f16(A2_16);
            U_jik += 3*dof; __builtin_prefetch(U_jik + 9*dof, 0, 0);
            x0_32 = vld1q_f32(x4); x4 += dof; __builtin_prefetch(x4 + 3*dof, 0, 0);
            x1_32 = vld1q_f32(x5); x5 += dof; __builtin_prefetch(x5 + 3*dof, 0, 0);
            x2_32 = vld1q_f32(x6); x6 += dof; __builtin_prefetch(x6 + 3*dof, 0, 0);
            tmp0 = vmlsq_f32(tmp0, A0_32, x0_32);
            tmp0 = vmlsq_f32(tmp0, A1_32, x1_32);
            tmp0 = vmlsq_f32(tmp0, A2_32, x2_32);// 此时tmp暂存了 b - 非本位的aj*xj
            
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            A0_2_16 = vld3_f16(invD_jik); invD_jik += dof*dof; __builtin_prefetch(invD_jik + 3*dof*dof, 0, 0);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            tmp0 = vld1q_f32(x_jik);// 此时tmp0暂存原来的x解
            r0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w*D^{-1}*(b - 非本位的aj*xj)
            float buf[4]; vst1q_f32(buf, r0);
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x_jik[f] = buf[f];
            
            x_jik += dof; __builtin_prefetch(x_jik + 3*dof,1);
        }
    }
}

template<int dof, int num_L_diag, int num_U_diag>
void AOS_compress_point_backward_ALL_3d_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * U_jik, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * dummy)
{
    static_assert(num_L_diag == 3);
    const float * x0 = x_jik - vec_dki_size, * x6 = x_jik + vec_dki_size,
                * x1 = x_jik - vec_dk_size , * x5 = x_jik + vec_dk_size ,
                * x2 = x_jik - dof         , * x4 = x_jik + dof         ;
    const float32x4_t vwgts = vdupq_n_f32(wgt), vone_minus_wgts = vdupq_n_f32(1.0 - wgt);
    if constexpr (dof == 3) {
        float16x4x3_t A0_2_16;
        float16x4_t A0_16, A1_16, A2_16;
        float32x4_t A0_32, A1_32, A2_32, x0_32, x1_32, x2_32, tmp0;
        float32x4_t c0, c1, c2, r0, r1, r2;
        for (int k = 0; k < num; k++) {
            b_jik -= dof; __builtin_prefetch(b_jik - 3*dof, 0, 0); tmp0 = vld1q_f32(b_jik);
            
            L_jik -= 3*dof; __builtin_prefetch(L_jik - 9*dof, 0, 0);
            A0_16 = vld1_f16(L_jik        ); A0_32 = vcvt_f32_f16(A0_16);
            A1_16 = vld1_f16(L_jik +   dof); A1_32 = vcvt_f32_f16(A1_16);
            A2_16 = vld1_f16(L_jik + 2*dof); A2_32 = vcvt_f32_f16(A2_16);
            x0 -= dof; __builtin_prefetch(x0 - 3*dof, 0, 0); x0_32 = vld1q_f32(x0);
            x1 -= dof; __builtin_prefetch(x1 - 3*dof, 0, 0); x1_32 = vld1q_f32(x1);
            x2 -= dof; __builtin_prefetch(x2 - 3*dof, 0, 0); x2_32 = vld1q_f32(x2);
            tmp0 = vmlsq_f32(tmp0, A0_32, x0_32);
            tmp0 = vmlsq_f32(tmp0, A1_32, x1_32);
            tmp0 = vmlsq_f32(tmp0, A2_32, x2_32);
            
            U_jik -= 3*dof; __builtin_prefetch(U_jik - 9*dof, 0, 0);
            A0_16 = vld1_f16(U_jik        ); A0_32 = vcvt_f32_f16(A0_16);
            A1_16 = vld1_f16(U_jik +   dof); A1_32 = vcvt_f32_f16(A1_16);
            A2_16 = vld1_f16(U_jik + 2*dof); A2_32 = vcvt_f32_f16(A2_16);
            x4 -= dof; __builtin_prefetch(x4 - 3*dof, 0, 0); x0_32 = vld1q_f32(x4);
            x5 -= dof; __builtin_prefetch(x5 - 3*dof, 0, 0); x1_32 = vld1q_f32(x5);
            x6 -= dof; __builtin_prefetch(x6 - 3*dof, 0, 0); x2_32 = vld1q_f32(x6);
            tmp0 = vmlsq_f32(tmp0, A0_32, x0_32);
            tmp0 = vmlsq_f32(tmp0, A1_32, x1_32);
            tmp0 = vmlsq_f32(tmp0, A2_32, x2_32);// 此时tmp暂存了 b - 非本位的aj*xj
            
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            invD_jik -= dof*dof; __builtin_prefetch(invD_jik - 3*dof*dof, 0, 0); A0_2_16 = vld3_f16(invD_jik);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            x_jik -= dof; __builtin_prefetch(x_jik - 3*dof,1); tmp0 = vld1q_f32(x_jik);// 此时tmp0暂存原来的x解
            tmp0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w*D^{-1}*(b - 非本位的aj*xj)

            float buf[4]; vst1q_f32(buf, tmp0);
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x_jik[f] = buf[f];
        }
    }
}

template<int dof, int num_L_diag>
void AOS_compress_point_forward_zero_3d_scaled_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * dummy0, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * sqD_jik)
{
    static_assert(num_L_diag == 3);
    const float * x0 = x_jik - vec_dki_size, * sqD0 = sqD_jik - vec_dki_size,
                * x1 = x_jik - vec_dk_size , * sqD1 = sqD_jik - vec_dk_size ,
                * x2 = x_jik - dof         , * sqD2 = sqD_jik - dof         ;
    const float32x4_t vwgts = vdupq_n_f32(wgt);
    if constexpr (dof == 3) {
        float16x4x3_t A0_2_16;
        float16x4_t A0_16, A1_16, A2_16;
        float32x4_t A0_32, A1_32, A2_32, x0_32, x1_32, x2_32, tmp0, myQ;
        float32x4_t c0, c1, c2, r0, r1, r2;
        for (int k = 0; k < num; k++) {
            tmp0 = vld1q_f32(b_jik); b_jik += dof; __builtin_prefetch(b_jik + 9*dof, 0, 0);
            myQ  = vld1q_f32(sqD_jik); sqD_jik += dof; __builtin_prefetch(sqD_jik + 9*dof, 0, 0);
            tmp0 = vdivq_f32(tmp0, myQ);// tmp0 此时暂存 Q^{-1/2}*b

            A0_16 = vld1_f16(L_jik        ); A0_32 = vcvt_f32_f16(A0_16);
            A1_16 = vld1_f16(L_jik +   dof); A1_32 = vcvt_f32_f16(A1_16);
            A2_16 = vld1_f16(L_jik + 2*dof); A2_32 = vcvt_f32_f16(A2_16);
            L_jik += 3*dof; __builtin_prefetch(L_jik + 27*dof, 0, 0);
            x0_32 = vld1q_f32(x0); x0 += dof; __builtin_prefetch(x0 + 9*dof, 0, 0);
            x1_32 = vld1q_f32(x1); x1 += dof; __builtin_prefetch(x1 + 9*dof, 0, 0);
            x2_32 = vld1q_f32(x2); x2 += dof; __builtin_prefetch(x2 + 9*dof, 0, 0);
            c0 = vld1q_f32(sqD0); sqD0 += dof; __builtin_prefetch(sqD0 + 9*dof, 0, 0);
            c1 = vld1q_f32(sqD1); sqD1 += dof; __builtin_prefetch(sqD1 + 9*dof, 0, 0);
            c2 = vld1q_f32(sqD2); sqD2 += dof; __builtin_prefetch(sqD2 + 9*dof, 0, 0);
            A0_32 = vmulq_f32(A0_32, c0);
            A1_32 = vmulq_f32(A1_32, c1);
            A2_32 = vmulq_f32(A2_32, c2);
            tmp0 = vmlsq_f32(tmp0, A0_32, x0_32);
            tmp0 = vmlsq_f32(tmp0, A1_32, x1_32);
            tmp0 = vmlsq_f32(tmp0, A2_32, x2_32);// 此时tmp暂存了 Q^{-1/2}*b - Lbar*Q^{1/2}*x
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)

            A0_2_16 = vld3_f16(invD_jik); invD_jik += dof*dof; __builtin_prefetch(invD_jik + 9*dof*dof, 0, 0);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)
            r0 = vdivq_f32(r0, myQ);// w * Q^{-1/2} * D^{-1} * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)

            float buf[4]; vst1q_f32(buf, r0);// 不能直接存，必须先倒一趟，以免向量寄存器内的lane3会污染原数据
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x_jik[f] = buf[f];

            x_jik += dof; __builtin_prefetch(x_jik + 9*dof,1);
        }
    }
}

template<int dof, int num_L_diag, int num_U_diag>
void AOS_compress_point_forward_ALL_3d_scaled_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * U_jik, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * sqD_jik)
{
    static_assert(num_L_diag == 3);
    const float * x0 = x_jik - vec_dki_size, * x6 = x_jik + vec_dki_size,
                * x1 = x_jik - vec_dk_size , * x5 = x_jik + vec_dk_size ,
                * x2 = x_jik - dof         , * x4 = x_jik + dof         ;
    const float * sqD0 = sqD_jik - vec_dki_size, * sqD6 = sqD_jik + vec_dki_size,
                * sqD1 = sqD_jik - vec_dk_size , * sqD5 = sqD_jik + vec_dk_size ,
                * sqD2 = sqD_jik - dof         , * sqD4 = sqD_jik + dof         ;
    const float32x4_t vwgts = vdupq_n_f32(wgt), vone_minus_wgts = vdupq_n_f32(1.0 - wgt);
    if constexpr (dof == 3) {
        float16x4x3_t A0_2_16;
        float16x4_t A0_16, A1_16, A2_16;
        float32x4_t A0_32, A1_32, A2_32, x0_32, x1_32, x2_32, tmp0, myQ;
        float32x4_t c0, c1, c2, r0, r1, r2;
        for (int k = 0; k < num; k++) {
            tmp0 = vld1q_f32(b_jik); b_jik += dof; __builtin_prefetch(b_jik + 9*dof, 0, 0);
            myQ  = vld1q_f32(sqD_jik); sqD_jik += dof; __builtin_prefetch(sqD_jik + 9*dof, 0, 0);
            tmp0 = vdivq_f32(tmp0, myQ);// tmp0 此时暂存 Q^{-1/2}*b

            A0_16 = vld1_f16(L_jik        ); A0_32 = vcvt_f32_f16(A0_16);
            A1_16 = vld1_f16(L_jik +   dof); A1_32 = vcvt_f32_f16(A1_16);
            A2_16 = vld1_f16(L_jik + 2*dof); A2_32 = vcvt_f32_f16(A2_16);
            L_jik += 3*dof; __builtin_prefetch(L_jik + 27*dof, 0, 0);
            x0_32 = vld1q_f32(x0); x0 += dof; __builtin_prefetch(x0 + 9*dof, 0, 0);
            x1_32 = vld1q_f32(x1); x1 += dof; __builtin_prefetch(x1 + 9*dof, 0, 0);
            x2_32 = vld1q_f32(x2); x2 += dof; __builtin_prefetch(x2 + 9*dof, 0, 0);
            c0 = vld1q_f32(sqD0); sqD0 += dof; __builtin_prefetch(sqD0 + 9*dof, 0, 0);
            c1 = vld1q_f32(sqD1); sqD1 += dof; __builtin_prefetch(sqD1 + 9*dof, 0, 0);
            c2 = vld1q_f32(sqD2); sqD2 += dof; __builtin_prefetch(sqD2 + 9*dof, 0, 0);
            A0_32 = vmulq_f32(A0_32, c0);
            A1_32 = vmulq_f32(A1_32, c1);
            A2_32 = vmulq_f32(A2_32, c2);
            tmp0 = vmlsq_f32(tmp0, A0_32, x0_32);
            tmp0 = vmlsq_f32(tmp0, A1_32, x1_32);
            tmp0 = vmlsq_f32(tmp0, A2_32, x2_32);
            
            A0_16 = vld1_f16(U_jik        ); A0_32 = vcvt_f32_f16(A0_16);
            A1_16 = vld1_f16(U_jik +   dof); A1_32 = vcvt_f32_f16(A1_16);
            A2_16 = vld1_f16(U_jik + 2*dof); A2_32 = vcvt_f32_f16(A2_16);
            U_jik += 3*dof; __builtin_prefetch(U_jik + 27*dof, 0, 0);
            x0_32 = vld1q_f32(x4); x4 += dof; __builtin_prefetch(x4 + 9*dof, 0, 0);
            x1_32 = vld1q_f32(x5); x5 += dof; __builtin_prefetch(x5 + 9*dof, 0, 0);
            x2_32 = vld1q_f32(x6); x6 += dof; __builtin_prefetch(x6 + 9*dof, 0, 0);
            c0 = vld1q_f32(sqD4); sqD4 += dof; __builtin_prefetch(sqD4 + 9*dof, 0, 0);
            c1 = vld1q_f32(sqD5); sqD5 += dof; __builtin_prefetch(sqD5 + 9*dof, 0, 0);
            c2 = vld1q_f32(sqD6); sqD6 += dof; __builtin_prefetch(sqD6 + 9*dof, 0, 0);
            A0_32 = vmulq_f32(A0_32, c0);
            A1_32 = vmulq_f32(A1_32, c1);
            A2_32 = vmulq_f32(A2_32, c2);
            tmp0 = vmlsq_f32(tmp0, A0_32, x0_32);
            tmp0 = vmlsq_f32(tmp0, A1_32, x1_32);
            tmp0 = vmlsq_f32(tmp0, A2_32, x2_32);// 此时tmp暂存了 Q^{-1/2}*b - Lbar*Q^{1/2}*x
            
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)

            A0_2_16 = vld3_f16(invD_jik); invD_jik += dof*dof; __builtin_prefetch(invD_jik + 9*dof*dof, 0, 0);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)
            r0 = vdivq_f32(r0, myQ);// r0 暂存 w * Q^{-1/2} * D^{-1} * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)

            tmp0 = vld1q_f32(x_jik);// 此时tmp0暂存原来的x解
            r0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w* Q^{-1/2} * D^{-1} * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)
            float buf[4]; vst1q_f32(buf, r0);
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x_jik[f] = buf[f];
            
            x_jik += dof; __builtin_prefetch(x_jik + 9*dof,1);
        }
    }
}

template<int dof, int num_L_diag, int num_U_diag>
void AOS_compress_point_backward_ALL_3d_scaled_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * U_jik, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * sqD_jik)
{
    static_assert(num_L_diag == 3);
    const float * x0 = x_jik - vec_dki_size, * x6 = x_jik + vec_dki_size,
                * x1 = x_jik - vec_dk_size , * x5 = x_jik + vec_dk_size ,
                * x2 = x_jik - dof         , * x4 = x_jik + dof         ;
    const float * sqD0 = sqD_jik - vec_dki_size, * sqD6 = sqD_jik + vec_dki_size,
                * sqD1 = sqD_jik - vec_dk_size , * sqD5 = sqD_jik + vec_dk_size ,
                * sqD2 = sqD_jik - dof         , * sqD4 = sqD_jik + dof         ;
    const float32x4_t vwgts = vdupq_n_f32(wgt), vone_minus_wgts = vdupq_n_f32(1.0 - wgt);
    if constexpr (dof == 3) {
        float16x4x3_t A0_2_16;
        float16x4_t A0_16, A1_16, A2_16;
        float32x4_t A0_32, A1_32, A2_32, x0_32, x1_32, x2_32, tmp0, myQ;
        float32x4_t c0, c1, c2, r0, r1, r2;
        for (int k = 0; k < num; k++) {
            b_jik -= dof; __builtin_prefetch(b_jik - 9*dof, 0, 0); tmp0 = vld1q_f32(b_jik);
            sqD_jik -= dof; __builtin_prefetch(sqD_jik - 9*dof, 0, 0); myQ = vld1q_f32(sqD_jik); 
            tmp0 = vdivq_f32(tmp0, myQ);// tmp0 此时暂存 Q^{-1/2}*b
            
            L_jik -= 3*dof; __builtin_prefetch(L_jik - 27*dof, 0, 0);
            A0_16 = vld1_f16(L_jik        ); A0_32 = vcvt_f32_f16(A0_16);
            A1_16 = vld1_f16(L_jik +   dof); A1_32 = vcvt_f32_f16(A1_16);
            A2_16 = vld1_f16(L_jik + 2*dof); A2_32 = vcvt_f32_f16(A2_16);
            x0 -= dof; __builtin_prefetch(x0 - 9*dof, 0, 0); x0_32 = vld1q_f32(x0);
            x1 -= dof; __builtin_prefetch(x1 - 9*dof, 0, 0); x1_32 = vld1q_f32(x1);
            x2 -= dof; __builtin_prefetch(x2 - 9*dof, 0, 0); x2_32 = vld1q_f32(x2);
            sqD0 -= dof; __builtin_prefetch(sqD0 - 9*dof, 0, 0); c0 = vld1q_f32(sqD0);
            sqD1 -= dof; __builtin_prefetch(sqD1 - 9*dof, 0, 0); c1 = vld1q_f32(sqD1);
            sqD2 -= dof; __builtin_prefetch(sqD2 - 9*dof, 0, 0); c2 = vld1q_f32(sqD2);
            A0_32 = vmulq_f32(A0_32, c0);
            A1_32 = vmulq_f32(A1_32, c1);
            A2_32 = vmulq_f32(A2_32, c2);
            tmp0 = vmlsq_f32(tmp0, A0_32, x0_32);
            tmp0 = vmlsq_f32(tmp0, A1_32, x1_32);
            tmp0 = vmlsq_f32(tmp0, A2_32, x2_32);
            
            U_jik -= 3*dof; __builtin_prefetch(U_jik - 27*dof, 0, 0);
            A0_16 = vld1_f16(U_jik        ); A0_32 = vcvt_f32_f16(A0_16);
            A1_16 = vld1_f16(U_jik +   dof); A1_32 = vcvt_f32_f16(A1_16);
            A2_16 = vld1_f16(U_jik + 2*dof); A2_32 = vcvt_f32_f16(A2_16);
            x4 -= dof; __builtin_prefetch(x4 - 9*dof, 0, 0); x0_32 = vld1q_f32(x4);
            x5 -= dof; __builtin_prefetch(x5 - 9*dof, 0, 0); x1_32 = vld1q_f32(x5);
            x6 -= dof; __builtin_prefetch(x6 - 9*dof, 0, 0); x2_32 = vld1q_f32(x6);
            sqD4 -= dof; __builtin_prefetch(sqD4 - 9*dof, 0, 0); c0 = vld1q_f32(sqD4);
            sqD5 -= dof; __builtin_prefetch(sqD5 - 9*dof, 0, 0); c1 = vld1q_f32(sqD5);
            sqD6 -= dof; __builtin_prefetch(sqD6 - 9*dof, 0, 0); c2 = vld1q_f32(sqD6);
            A0_32 = vmulq_f32(A0_32, c0);
            A1_32 = vmulq_f32(A1_32, c1);
            A2_32 = vmulq_f32(A2_32, c2);
            tmp0 = vmlsq_f32(tmp0, A0_32, x0_32);
            tmp0 = vmlsq_f32(tmp0, A1_32, x1_32);
            tmp0 = vmlsq_f32(tmp0, A2_32, x2_32);// 此时tmp暂存了 Q^{-1/2}*b - Lbar*Q^{1/2}*x
            
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)

            invD_jik -= dof*dof; __builtin_prefetch(invD_jik - 9*dof*dof, 0, 0); A0_2_16 = vld3_f16(invD_jik);
            c0 = vcvt_f32_f16(A0_2_16.val[0])            ; c1 = vcvt_f32_f16(A0_2_16.val[1])            ; c2 = vcvt_f32_f16(A0_2_16.val[2])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1)); r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2));
            r0 = vaddq_f32(r0, r1); r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)
            r0 = vdivq_f32(r0, myQ);// r0 暂存 w * Q^{-1/2} * D^{-1} * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)

            x_jik -= dof; __builtin_prefetch(x_jik - 9*dof,1); tmp0 = vld1q_f32(x_jik);// 此时tmp0暂存原来的x解
            tmp0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w * Q^{-1/2} * D^{-1} * (Q^{-1/2}*b - Lbar*Q^{1/2}*x)

            float buf[4]; vst1q_f32(buf, tmp0);
            #pragma GCC unroll (4)
            for (int f = 0; f < dof; f++)
                x_jik[f] = buf[f];
        }
    }
}



#endif