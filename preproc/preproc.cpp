/*
    根据谢炎提供的CSR文件，预处理成若干条对角线/带
*/
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
using namespace std;

typedef long long idx_t;
typedef double data_t;
const idx_t num_diag = 15;
const idx_t num_dof = 3;
const idx_t stencil_offset[num_diag * num_dof] = {
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

void calc_ijk(idx_t csr_idx, idx_t nx, idx_t ny, idx_t nz, idx_t dof, idx_t * i, idx_t * j, idx_t * k, idx_t * var) {
    idx_t idx = csr_idx / dof;
    *var = csr_idx - idx * dof;

    *j =  idx / (nx * nz);
    *i = (idx - *j * nx * nz) / nz;
    *k =  idx - *j * nx * nz - *i * nz;
}

int main(int argc, char * argv[])
{
    const char * mat_name = argv[1];
    const char * rhs_name = argv[2];
    const idx_t nx = atoi(argv[3]);
    const idx_t ny = atoi(argv[4]);
    const idx_t nz = atoi(argv[5]);
    const idx_t tot_elems = nx * ny * nz;
    printf("nx %lld ny %lld nz %lld\n", nx, ny, nz);
    
    FILE * fp = fopen(mat_name, "r");
    idx_t nrows, ncols, nnz;
    fscanf(fp, "%lld %lld %lld", &nrows, &ncols, &nnz);
    assert(nrows == ncols && nrows == num_dof * tot_elems);
    assert(nnz > 0);

    // 读入CSR矩阵
    idx_t * row_ptr = new idx_t [nrows + 1];
    idx_t * col_idx = new idx_t [nnz];
    data_t* vals    = new data_t[nnz];
    data_t dummy;
    for (idx_t i = 0; i < nrows + 1; i++)
        fscanf(fp, "%lld", row_ptr + i);
    for (idx_t j = 0; j < nnz; j++)
        fscanf(fp, "%lld", col_idx + j);
    for (idx_t j = 0; j < nnz; j++)
        fscanf(fp, "%lf", vals + j);
    assert(fscanf(fp, "%lf", &dummy) == EOF);
    fclose(fp);

    data_t * diag_SOA = new data_t [tot_elems * num_diag * num_dof * num_dof];
    for (idx_t p = 0; p < tot_elems * num_diag * num_dof * num_dof; p++)// 先置零
        diag_SOA[p] = 0.0;

    for (idx_t row = 0; row < nrows; row++) {
        idx_t i, j, k, vid;
        calc_ijk(row, nx, ny, nz, num_dof, &i, &j, &k, &vid);
        idx_t elem_id = row / num_dof;
        idx_t pbeg = row_ptr[row], pend = row_ptr[row+1];
        for (idx_t p = pbeg; p < pend; p++) {
            idx_t col = col_idx[p];
            idx_t ngb_i, ngb_j, ngb_k, ngb_vid;
            calc_ijk(col, nx, ny, nz, num_dof, &ngb_i, &ngb_j, &ngb_k, &ngb_vid);
            idx_t tgt_d = -1;// target direction
            for (idx_t d = 0; d < num_diag; d++) {// 确定属于哪个方向
                if (stencil_offset[d*3  ] + j == ngb_j &&
                    stencil_offset[d*3+1] + i == ngb_i &&
                    stencil_offset[d*3+2] + k == ngb_k) {
                    tgt_d = d;
                    break;
                }
            }
            // assert(tgt_d >= 0);
            if (tgt_d < 0) {
                printf("row %lld col %lld\n", row, col);
                printf("(i,j,k)=(%lld,%lld,%lld), (ni,nj,nk)=(%lld,%lld,%lld)\n", i, j, k, ngb_i, ngb_j, ngb_k);
                exit(1);
            }
            diag_SOA[(tot_elems * tgt_d + elem_id) * num_dof * num_dof + vid * num_dof + ngb_vid] = vals[p];
        }
    }
    
    for (idx_t id = 0; id < num_diag; id++) {
        char buf[100];
        sprintf(buf, "array_a.%lld", id);
        fp = fopen(buf, "wb+");
        idx_t size = fwrite(diag_SOA + id * tot_elems * num_dof * num_dof, sizeof(data_t), tot_elems * num_dof * num_dof, fp);
        assert(size == tot_elems * num_dof * num_dof);
        fclose(fp);
    }

    data_t * rhs = new data_t [tot_elems * num_dof];
    {
        fp = fopen(rhs_name, "r");
        idx_t num;
        fscanf(fp, "%lld", &num);
        assert(num == nrows);
        data_t check = 0.0;
        for (idx_t i = 0; i < nrows; i++) {
            fscanf(fp, "%lf", rhs + i);
            check += rhs[i] * rhs[i];
        }
        fclose(fp);
        printf("(b, b) = %.10e\n", check);
    }
    {
        fp = fopen("array_b", "wb+");
        idx_t size = fwrite(rhs, sizeof(data_t), tot_elems * num_dof, fp);
        assert(size == tot_elems * num_dof);
        fclose(fp);
    }

    data_t * test = new data_t [nrows];
    for (idx_t i = 0; i < nrows; i++) {
        data_t tmp = 0.0;
        idx_t pbeg = row_ptr[i], pend = row_ptr[i+1];
        for (idx_t p = pbeg; p < pend; p++)
            tmp += vals[p] * rhs[col_idx[p]];
        test[i] = tmp;
    }
    {// check (A*b, A*b)
        data_t check = 0.0;
        for (idx_t i = 0; i < nrows; i++)
            check += test[i] * test[i];
        printf("(A*b, A*b) = %.10e\n", check);
    }

    delete row_ptr; delete col_idx; delete vals;
    delete diag_SOA;
    delete rhs;
    return 0;
}