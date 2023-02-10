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
    constexpr int mat_prft = elms * num_diag;
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
    else if constexpr (dof == 4) {
        float16x4x4_t A_16;
        float32x4_t c0, c1, c2, c3;
        for (int k = 0; k < num; k++) {
            const __fp16 * aos_ptr = A_jik;
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0),
                        r2 = vdupq_n_f32(0.0), r3 = vdupq_n_f32(0.0);// 第几列的结果暂存
            for (int d = 0; d < num_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + mat_prft, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);

                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                xp += dof; __builtin_prefetch(xp + dof, 0);

                aos_ptr += elms;
            }
            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);
            vst1q_f32(y_jik, r0); y_jik += dof; __builtin_prefetch(y_jik + dof,1);
            A_jik += num_diag * elms;
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
    else if constexpr (dof == 4) {
        float16x4x4_t A_16;
        float32x4_t c0, c1, c2, c3, tmp0;
        for (int k = 0; k < num; k++) {
            const __fp16 * aos_ptr = L_jik;
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0),
                        r2 = vdupq_n_f32(0.0), r3 = vdupq_n_f32(0.0);// 第几列的结果暂存
            tmp0 = vld1q_f32(b_jik); b_jik += dof; __builtin_prefetch(b_jik + 2*dof, 0, 0);

            for (int d = 0; d < num_L_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 2*elms * num_L_diag, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            L_jik += num_L_diag * elms;

            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);
            tmp0 = vsubq_f32(tmp0, r0);// 此时tmp暂存了 b - 非本位的aj*xj
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            A_16 = vld4_f16(invD_jik); invD_jik += elms; __builtin_prefetch(invD_jik + 2*elms, 0, 0);
            c0 = vcvt_f32_f16(A_16.val[0])            ; c1 = vcvt_f32_f16(A_16.val[1])            ;
            c2 = vcvt_f32_f16(A_16.val[2])            ; c3 = vcvt_f32_f16(A_16.val[3])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1));
            r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2)); r3 = vmulq_n_f32(c3, vgetq_lane_f32(tmp0, 3));
            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            vst1q_f32(x_jik, r0); x_jik += dof; __builtin_prefetch(x_jik + 2*dof,1);
        }
    }
}

template<int dof, int num_L_diag>
void AOS_point_forward_zero_3d_irr_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * dummy0, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * dummy1,
    const int irr_k, const float * contrib)
{
    constexpr int elms = dof*dof;
    const float * xa[num_L_diag];
    pgsf_prepare_xa;
    const float32x4_t vwgts = vdupq_n_f32(wgt), irr_ctrb = vld1q_f32(contrib);
    if constexpr (dof == 4) {
        float16x4x4_t A_16;
        float32x4_t c0, c1, c2, c3, tmp0;
        for (int k = 0; k < num; k++) {
            const __fp16 * aos_ptr = L_jik;
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0),
                        r2 = vdupq_n_f32(0.0), r3 = vdupq_n_f32(0.0);// 第几列的结果暂存
            tmp0 = vld1q_f32(b_jik); b_jik += dof; __builtin_prefetch(b_jik + 2*dof, 0, 0);
            if (k == irr_k) tmp0 = vsubq_f32(tmp0, irr_ctrb);

            for (int d = 0; d < num_L_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 2*elms * num_L_diag, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            L_jik += num_L_diag * elms;

            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);
            tmp0 = vsubq_f32(tmp0, r0);// 此时tmp暂存了 b - 非本位的aj*xj
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            A_16 = vld4_f16(invD_jik); invD_jik += elms; __builtin_prefetch(invD_jik + 2*elms, 0, 0);
            c0 = vcvt_f32_f16(A_16.val[0])            ; c1 = vcvt_f32_f16(A_16.val[1])            ;
            c2 = vcvt_f32_f16(A_16.val[2])            ; c3 = vcvt_f32_f16(A_16.val[3])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1));
            r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2)); r3 = vmulq_n_f32(c3, vgetq_lane_f32(tmp0, 3));
            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            vst1q_f32(x_jik, r0); x_jik += dof; __builtin_prefetch(x_jik + 2*dof,1);
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
            
            x_jik += dof; __builtin_prefetch(x_jik + 2*dof,1);
        }
    }
    else if constexpr (dof == 4) {
        const __fp16 * aos_ptr = nullptr;
        float16x4x4_t A_16;
        float32x4_t c0, c1, c2, c3, tmp0;
        for (int k = 0; k < num; k++) {
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0),
                        r2 = vdupq_n_f32(0.0), r3 = vdupq_n_f32(0.0);// 第几列的结果暂存
            tmp0 = vld1q_f32(b_jik); b_jik += dof; __builtin_prefetch(b_jik + 2*dof, 0, 0);

            aos_ptr = L_jik;// L部分
            for (int d = 0; d < num_L_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 2*elms * num_L_diag, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            L_jik += num_L_diag * elms;

            aos_ptr = U_jik;// U部分
            for (int d = num_L_diag + 1; d < num_L_diag + 1 + num_U_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 2*elms * num_U_diag, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            U_jik += num_U_diag * elms;

            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);
            tmp0 = vsubq_f32(tmp0, r0);// 此时tmp暂存了 b - 非本位的aj*xj
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            A_16 = vld4_f16(invD_jik); invD_jik += elms; __builtin_prefetch(invD_jik + 2*elms, 0, 0);
            c0 = vcvt_f32_f16(A_16.val[0])            ; c1 = vcvt_f32_f16(A_16.val[1])            ;
            c2 = vcvt_f32_f16(A_16.val[2])            ; c3 = vcvt_f32_f16(A_16.val[3])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1));
            r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2)); r3 = vmulq_n_f32(c3, vgetq_lane_f32(tmp0, 3));
            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            tmp0 = vld1q_f32(x_jik);// 此时tmp0暂存原来的x解
            r0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w*D^{-1}*(b - 非本位的aj*xj)
            
            vst1q_f32(x_jik, r0); x_jik += dof; __builtin_prefetch(x_jik + 2*dof,1);
        }
    }
}


template<int dof, int num_L_diag, int num_U_diag>
void AOS_point_forward_ALL_3d_irr_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * U_jik, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * dummy,
    const int irr_k, const float * contrib)
{
    constexpr int elms = dof*dof;
    const float * xa[num_L_diag + 1 + num_U_diag];
    static_assert(num_L_diag == num_U_diag);
    pgsAll_prepare_xa;
    const float32x4_t vwgts = vdupq_n_f32(wgt), vone_minus_wgts = vdupq_n_f32(1.0 - wgt), irr_ctrb = vld1q_f32(contrib);
    if constexpr (dof == 4) {
        const __fp16 * aos_ptr = nullptr;
        float16x4x4_t A_16;
        float32x4_t c0, c1, c2, c3, tmp0;
        for (int k = 0; k < num; k++) {
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0),
                        r2 = vdupq_n_f32(0.0), r3 = vdupq_n_f32(0.0);// 第几列的结果暂存
            tmp0 = vld1q_f32(b_jik); b_jik += dof; __builtin_prefetch(b_jik + 2*dof, 0, 0);
            if (k == irr_k) tmp0 = vsubq_f32(tmp0, irr_ctrb);

            aos_ptr = L_jik;// L部分
            for (int d = 0; d < num_L_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 2*elms * num_L_diag, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            L_jik += num_L_diag * elms;

            aos_ptr = U_jik;// U部分
            for (int d = num_L_diag + 1; d < num_L_diag + 1 + num_U_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr + 2*elms * num_U_diag, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                xp += dof; __builtin_prefetch(xp + 2*dof, 0);

                aos_ptr += elms;
            }
            U_jik += num_U_diag * elms;

            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);
            tmp0 = vsubq_f32(tmp0, r0);// 此时tmp暂存了 b - 非本位的aj*xj
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            A_16 = vld4_f16(invD_jik); invD_jik += elms; __builtin_prefetch(invD_jik + 2*elms, 0, 0);
            c0 = vcvt_f32_f16(A_16.val[0])            ; c1 = vcvt_f32_f16(A_16.val[1])            ;
            c2 = vcvt_f32_f16(A_16.val[2])            ; c3 = vcvt_f32_f16(A_16.val[3])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1));
            r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2)); r3 = vmulq_n_f32(c3, vgetq_lane_f32(tmp0, 3));
            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            tmp0 = vld1q_f32(x_jik);// 此时tmp0暂存原来的x解
            r0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w*D^{-1}*(b - 非本位的aj*xj)
            
            vst1q_f32(x_jik, r0); x_jik += dof; __builtin_prefetch(x_jik + 2*dof,1);
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
    else if constexpr (dof == 4) {
        const __fp16 * aos_ptr = nullptr;
        float16x4x4_t A_16;
        float32x4_t c0, c1, c2, c3, tmp0;
        for (int k = 0; k < num; k++) {// 做完剩下的元素
            b_jik -= dof; __builtin_prefetch(b_jik - 2*dof, 0, 0); tmp0 = vld1q_f32(b_jik);
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0),
                        r2 = vdupq_n_f32(0.0), r3 = vdupq_n_f32(0.0);// 第几列的结果暂存

            L_jik -= num_L_diag * elms; aos_ptr = L_jik;
            for (int d = 0; d < num_L_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr - 2*elms * num_L_diag, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);
                xp -= dof; __builtin_prefetch(xp - 2*dof, 0);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                aos_ptr += elms;
            }

            U_jik -= num_U_diag * elms; aos_ptr = U_jik;
            for (int d = num_L_diag + 1; d < num_L_diag + 1 + num_U_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr - 2*elms * num_U_diag, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);
                xp -= dof; __builtin_prefetch(xp - 2*dof, 0);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                aos_ptr += elms;
            }

            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);
            tmp0 = vsubq_f32(tmp0, r0);// 此时tmp暂存了 b - 非本位的aj*xj
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            invD_jik -= elms;
            A_16 = vld4_f16(invD_jik); __builtin_prefetch(invD_jik - 2*elms, 0, 0);
            c0 = vcvt_f32_f16(A_16.val[0])            ; c1 = vcvt_f32_f16(A_16.val[1])            ;
            c2 = vcvt_f32_f16(A_16.val[2])            ; c3 = vcvt_f32_f16(A_16.val[3])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1));
            r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2)); r3 = vmulq_n_f32(c3, vgetq_lane_f32(tmp0, 3));
            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            x_jik -= dof; __builtin_prefetch(x_jik - 2*dof,1); tmp0 = vld1q_f32(x_jik);// 此时tmp0暂存原来的x解
            r0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w*D^{-1}*(b - 非本位的aj*xj)
            vst1q_f32(x_jik, r0);
        }
    }
}


template<int dof, int num_L_diag, int num_U_diag>
void AOS_point_backward_ALL_3d_irr_Cal32Stg16(const int num,
    const int vec_dk_size, const int vec_dki_size, const float wgt,
    const __fp16 * L_jik, const __fp16 * U_jik, const __fp16 * invD_jik,
    const float * b_jik, float * x_jik, const float * dummy,
    const int irr_k, const float * contrib)
{
    constexpr int elms = dof*dof;
    const float * xa[num_L_diag + 1 + num_U_diag];
    static_assert(num_L_diag == num_U_diag);
    pgsAll_prepare_xa;
    const float32x4_t vwgts = vdupq_n_f32(wgt), vone_minus_wgts = vdupq_n_f32(1.0 - wgt), irr_ctrb = vld1q_f32(contrib);
    if constexpr (dof == 4) {
        const __fp16 * aos_ptr = nullptr;
        float16x4x4_t A_16;
        float32x4_t c0, c1, c2, c3, tmp0;
        for (int k = 0; k < num; k++) {// 做完剩下的元素
            b_jik -= dof; __builtin_prefetch(b_jik - 2*dof, 0, 0); tmp0 = vld1q_f32(b_jik);
            float32x4_t r0 = vdupq_n_f32(0.0), r1 = vdupq_n_f32(0.0),
                        r2 = vdupq_n_f32(0.0), r3 = vdupq_n_f32(0.0);// 第几列的结果暂存
            if (k == num - 1 - irr_k) tmp0 = vsubq_f32(tmp0, irr_ctrb);

            L_jik -= num_L_diag * elms; aos_ptr = L_jik;
            for (int d = 0; d < num_L_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr - 2*elms * num_L_diag, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);
                xp -= dof; __builtin_prefetch(xp - 2*dof, 0);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                aos_ptr += elms;
            }

            U_jik -= num_U_diag * elms; aos_ptr = U_jik;
            for (int d = num_L_diag + 1; d < num_L_diag + 1 + num_U_diag; d++) {
                const float * & xp = xa[d];// 取引用！
                A_16 = vld4_f16(aos_ptr      ); __builtin_prefetch(aos_ptr - 2*elms * num_U_diag, 0, 0);
                c0 = vcvt_f32_f16(A_16.val[0]); c1 = vcvt_f32_f16(A_16.val[1]);
                c2 = vcvt_f32_f16(A_16.val[2]); c3 = vcvt_f32_f16(A_16.val[3]);
                xp -= dof; __builtin_prefetch(xp - 2*dof, 0);
                r0 = vmlaq_n_f32(r0, c0, xp[0])  ; r1 = vmlaq_n_f32(r1, c1, xp[1])  ;
                r2 = vmlaq_n_f32(r2, c2, xp[2])  ; r3 = vmlaq_n_f32(r3, c3, xp[3])  ;
                aos_ptr += elms;
            }

            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);
            tmp0 = vsubq_f32(tmp0, r0);// 此时tmp暂存了 b - 非本位的aj*xj
            tmp0 = vmulq_f32(vwgts, tmp0);// w * (b - 非本位的aj*xj)

            invD_jik -= elms;
            A_16 = vld4_f16(invD_jik); __builtin_prefetch(invD_jik - 2*elms, 0, 0);
            c0 = vcvt_f32_f16(A_16.val[0])            ; c1 = vcvt_f32_f16(A_16.val[1])            ;
            c2 = vcvt_f32_f16(A_16.val[2])            ; c3 = vcvt_f32_f16(A_16.val[3])            ;
            r0 = vmulq_n_f32(c0, vgetq_lane_f32(tmp0, 0)); r1 = vmulq_n_f32(c1, vgetq_lane_f32(tmp0, 1));
            r2 = vmulq_n_f32(c2, vgetq_lane_f32(tmp0, 2)); r3 = vmulq_n_f32(c3, vgetq_lane_f32(tmp0, 3));
            r0 = vaddq_f32(r0, r1); r2 = vaddq_f32(r2, r3);
            r0 = vaddq_f32(r0, r2);// r0 暂存 w * D^{-1} * (b - 非本位的aj*xj)

            x_jik -= dof; __builtin_prefetch(x_jik - 2*dof,1); tmp0 = vld1q_f32(x_jik);// 此时tmp0暂存原来的x解
            r0 = vmlaq_f32(r0, vone_minus_wgts, tmp0);// x^{t+1} = (1-w)*x^{t} + w*D^{-1}*(b - 非本位的aj*xj)
            vst1q_f32(x_jik, r0);
        }
    }
}

#endif