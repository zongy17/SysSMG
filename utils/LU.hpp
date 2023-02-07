#ifndef SOLID_UTILS_LU_HPP
#define SOLID_UTILS_LU_HPP

#include "common.hpp"

#define L_IDX(i, j) ( (((i)*(  (i) - 1 )   ) >> 1) + (j)       )
#define U_IDX(i, j) ( (((i)*(2*N - (i) + 1)) >> 1) + (j) - (i) )

template<typename idx_t, typename data_t>
void dense_LU_decomp(const data_t * a, data_t * l, data_t * u, const idx_t num_rows, const idx_t num_cols)
{
    assert(num_rows == num_cols);
    const idx_t N = num_rows;
    // assume: a为N*N的二维数组（一维存储），l为1+2+...+N-1=N*(N-1)/2的长度的数组，u为1+2+...+N=(N+1)*N/2的长度的数组

    data_t (*A)[N] = (data_t (*)[N]) a;

    data_t rbuf[N];
    // float rbuf[N];
    for (idx_t i = 0; i < N; i++) {
        // 拷入当前行的值
        for (idx_t j = 0; j < N; j++)
            rbuf[j] = A[i][j];
        // 逐个消去左边的元素
        for (idx_t lj = 0; lj < i; lj++) {
            if (rbuf[lj] != 0.0) {
                rbuf[lj] /= u[U_IDX(lj, lj)];
                // rbuf[lj] = rbuf[lj] / (float) u[U_IDX(lj, lj)];
                // 欲消该元会对该元右边元素产生影响
                for (idx_t rj = lj + 1; rj < N; rj++) {
                    if (u[U_IDX(lj, rj)] != 0.0)
                        rbuf[rj] -= rbuf[lj] * u[U_IDX(lj, rj)];
                        // rbuf[rj] -= (float) (rbuf[lj] * (float) u[U_IDX(lj, rj)]);
                }
            }
        }
        // 拷回到L和U中
        for (idx_t j = 0; j < i; j++)
            l[L_IDX(i, j)] = rbuf[j];
        for (idx_t j = i; j < N; j++)
            u[U_IDX(i, j)] = rbuf[j];
    }
}

template<typename idx_t, typename data_t>
void dense_forward(const data_t * l, const data_t * rhs, data_t * x, const idx_t num_rows, const idx_t num_cols)
{
    assert(num_rows == num_cols);
    const idx_t N = num_rows;
    // assume：l为1+2+...+N-1=N*(N-1)/2的长度的数组
    for (idx_t i = 0; i < N; i++) {
        data_t res = rhs[i];
        for (idx_t j = 0; j < i; j++)
            res -= l[L_IDX(i, j)] * x[j];
        x[i] = res;
    }
}

template<typename idx_t, typename data_t>
void dense_backward(const data_t * u, const data_t * rhs, data_t * x, const idx_t num_rows, const idx_t num_cols)
{
    assert(num_rows == num_cols);
    const idx_t N = num_rows;
    // assume：u为1+2+...+N=(N+1)*N/2的长度的数组
    for (idx_t i = N - 1; i >= 0; i--) {
        data_t diag = u[U_IDX(i, i)];
        data_t res  = rhs[i];
        for (idx_t j = i + 1; j < N; j++)
            res -= u[U_IDX(i, j)] * x[j];
        x[i] = res / diag;
    }
}

#undef L_IDX
#undef U_IDX

template<typename idx_t, typename data_t, int dof>
inline void LU_outofplace(const data_t * A, data_t * L, data_t * U)
{
    memset(L, 0.0, sizeof(data_t) * dof * dof);
    memset(U, 0.0, sizeof(data_t) * dof * dof);
    data_t rbuf[dof];
    for (idx_t i = 0; i < dof; i++) {
        // 拷入当前行的值
        for (idx_t j = 0; j < dof; j++)
            rbuf[j] = A[i * dof + j];
        // 逐个消去左边的元素
        for (idx_t lj = 0; lj < i; lj++) {
            if (rbuf[lj] != 0.0) {
                rbuf[lj] /= U[lj * dof + lj];
                // 欲消该元会对该元右边元素产生影响
                for (idx_t rj = lj + 1; rj < dof; rj++) {
                    if (U[lj * dof + rj] != 0.0)
                        rbuf[rj] -= rbuf[lj] * U[lj * dof + rj];
                }
            }
        }
        // 拷回到L和U中
        for (idx_t j = 0; j < i; j++)
            L[i * dof + j] = rbuf[j];
        L[i * dof + i] = 1.0;// L的对角元
        for (idx_t j = i; j < dof; j++)
            U[i * dof + j] = rbuf[j];
    }
}

#endif