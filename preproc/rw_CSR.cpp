/*
    根据谢炎提供的CSR文件，预处理成若干条对角线/带
*/
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
using namespace std;

typedef long long idx_t;
// typedef int idx_t;
typedef double data_t;

int main(int argc, char * argv[])
{
    const char * mat_name = argv[1];
    const char * rhs_name = argv[2];

    FILE * fp = nullptr;
    
    fp = fopen(mat_name, "r");
    idx_t nrows, ncols, nnz;
    fscanf(fp, "%lld %lld %lld", &nrows, &ncols, &nnz);
    printf("nrows %lld ncols %lld nnz %lld\n", nrows, ncols, nnz);
    // fscanf(fp, "%d %d %d", &nrows, &ncols, &nnz);
    // printf("nrows %d ncols %d nnz %d\n", nrows, ncols, nnz);
    assert(nrows == ncols);
    assert(nnz > 0);

    // 读入CSR矩阵
    {
        idx_t * row_ptr = new idx_t [nrows + 1];
        for (idx_t i = 0; i < nrows + 1; i++)
            fscanf(fp, "%lld", row_ptr + i);
        assert(row_ptr[nrows] == nnz);
        printf("writing Ai\n");
        FILE * write_ptr = fopen("Ai.bin", "wb+");
        fwrite(row_ptr, sizeof(idx_t), nrows + 1, write_ptr);
        fclose(write_ptr);
        delete row_ptr;
    }
    {
        idx_t * col_idx = new idx_t [nnz];
        for (idx_t i = 0; i < nnz; i++)
            fscanf(fp, "%lld", col_idx + i);
        printf("writing Aj\n");
        FILE * write_ptr = fopen("Aj.bin", "wb+");
        fwrite(col_idx, sizeof(idx_t), nnz, write_ptr);
        fclose(write_ptr);
        delete col_idx;
    }
    {
        data_t * vals = new data_t[nnz];
        for (idx_t i = 0; i < nnz; i++)
            fscanf(fp, "%lf", vals + i);
        printf("writing Av\n");
        FILE * write_ptr = fopen("Av.bin", "wb+");
        fwrite(vals, sizeof(data_t), nnz, write_ptr);
        fclose(write_ptr);
        delete vals;
    }

    double dummy;
    assert(fscanf(fp, "%lf", &dummy) == EOF);

    fclose(fp);

    fp = fopen(rhs_name, "r");
    {
        idx_t num;
        fscanf(fp, "%lld", &num);
        // fscanf(fp, "%d", &num);
        assert(num == nrows);
        data_t * b = new data_t [nrows];
        for (idx_t i = 0; i < nrows; i++)
            fscanf(fp, "%lf", b + i);
        printf("writing b\n");
        FILE * write_ptr = fopen("b.bin", "wb+");
        fwrite(b, sizeof(data_t), nrows, write_ptr);
        fclose(write_ptr);

        double dot = 0.0;
        for (idx_t i = 0; i < nrows; i++)
            dot += b[i] * b[i];
        printf("(b, b) = %.10e\n", dot);
        delete b;
    }
    fclose(fp);

    {
        data_t * x = new data_t [nrows];
        for (idx_t i = 0; i < nrows; i++)
            x[i] = 0.0;
        printf("writing x0\n");
        FILE * write_ptr = fopen("x0.bin", "wb+");
        fwrite(x, sizeof(data_t), nrows, write_ptr);
        fclose(write_ptr);
        delete x;
    }

    return 0;
}