#ifndef SOLID_DENSELU_HPP
#define SOLID_DENSELU_HPP

#include "../utils/par_struct_mat.hpp"
#include "../utils/LU.hpp"
#define USE_DENSE
#ifndef USE_DENSE
#ifdef __x86_64__
#include "mkl_types.h"
#include "mkl_pardiso.h"
#elif defined(__aarch64__)
// #include "hpdss.h"
#include "slu_ddefs.h"
#endif
#endif

typedef enum {DenseLU_3D7, DenseLU_3D27} DenseLU_type;

#ifndef USE_DENSE
template<typename idx_t, typename data_t, typename res_t>
class CSR_sparseMat {
public:
    idx_t nrows = 0;
    idx_t * row_ptr = nullptr;
    idx_t * col_idx = nullptr;
    data_t* vals = nullptr;
#ifdef __aarch64__
    char           equed[1];
    SuperMatrix    A, L, U;
    SuperMatrix    B, X;
    NCformat       *Astore;
    NCformat       *Ustore;
    SCformat       *Lstore;
    GlobalLU_t	   Glu; // facilitate multiple factorizations with SamePattern_SameRowPerm       
    double         *a;
    int            *asub, *xa;
    int            *perm_c; // column permutation vector
    int            *perm_r; // row permutations from partial pivoting
    int            *etree;
    void           *work = NULL;
    int            info, lwork = 0, nrhs, ldx;
    double         *b_buf, *x_buf;
    double         *R, *C;
    double         *ferr, *berr;
    double         u, rpg, rcond;
    mem_usage_t    mem_usage;
    superlu_options_t options;
    SuperLUStat_t stat;
#elif defined(__x86_64__)
    MKL_INT mtype = 1;
    MKL_INT nrhs = 1;
    void * pt[64];// internal solver memory pointer
    MKL_INT iparm[64];// pardiso control parameters
    MKL_INT maxfct, mnum, phase, error, msglvl;
    double ddum;// Double dummy
    MKL_INT idum;// Integer dummy.
    res_t * b_buf = nullptr;
#endif
    
    CSR_sparseMat(idx_t m, idx_t * Ai, idx_t * Aj, data_t * Av):
        nrows(m), row_ptr(Ai), col_idx(Aj), vals(Av) {  }
    ~CSR_sparseMat() {
#ifdef __aarch64__
        // if (row_ptr) delete row_ptr;
        // if (col_idx) delete col_idx;
        // if (vals   ) delete vals;
        SUPERLU_FREE (b_buf);
        SUPERLU_FREE (x_buf);
        SUPERLU_FREE (etree);
        SUPERLU_FREE (perm_r);
        SUPERLU_FREE (perm_c);
        SUPERLU_FREE (R);
        SUPERLU_FREE (C);
        SUPERLU_FREE (ferr);
        SUPERLU_FREE (berr);
        Destroy_CompRow_Matrix(&A);
        Destroy_SuperMatrix_Store(&B);
        Destroy_SuperMatrix_Store(&X);
        if ( lwork == 0 ) {
            Destroy_SuperNode_Matrix(&L);
            Destroy_CompRow_Matrix(&U);
        } else if ( lwork > 0 ) {
            SUPERLU_FREE(work);
        }
#elif defined(__x86_64__)
        if (row_ptr) delete row_ptr;
        if (col_idx) delete col_idx;
        if (vals   ) delete vals;
        if (b_buf) delete b_buf;
        phase = -1;// Release internal memory.
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                &nrows, &ddum, row_ptr, col_idx, &idum, &nrhs,
                iparm, &msglvl, &ddum, &ddum, &error);
#endif
    }
    double decomp() {
        assert(row_ptr && col_idx && vals);
        double t = 0.0;
#ifdef __aarch64__
        // Call Wang Xinliang's package here
        // int rtn;
        // hpdss::Analyse
        // rtn = handler.AnalyseFromCSR(nrows, row_ptr, col_idx, vals, &config);

        superlu_options_t options;
        set_default_options(&options);
        dCreate_CompRow_Matrix(&A, nrows, nrows, row_ptr[nrows], 
            vals, col_idx, row_ptr, SLU_NR, SLU_D, SLU_GE);
        Astore = (NCformat*) A.Store;

        const idx_t nrhs = 1;
        b_buf = new res_t [nrows];
        x_buf = new res_t [nrows];
        dCreate_Dense_Matrix(&B, nrows, nrhs, b_buf, nrows, SLU_DN, SLU_D, SLU_GE);
        dCreate_Dense_Matrix(&X, nrows, nrhs, x_buf, nrows, SLU_DN, SLU_D, SLU_GE);
        
        etree = new idx_t [nrows];
        perm_r= new idx_t [nrows];
        perm_c= new idx_t [nrows];
        if ( !(R = (double *) SUPERLU_MALLOC(A.nrow * sizeof(double))) ) 
            ABORT("SUPERLU_MALLOC fails for R[].");
        if ( !(C = (double *) SUPERLU_MALLOC(A.ncol * sizeof(double))) )
            ABORT("SUPERLU_MALLOC fails for C[].");
        if ( !(ferr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
            ABORT("SUPERLU_MALLOC fails for ferr[].");
        if ( !(berr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) ) 
            ABORT("SUPERLU_MALLOC fails for berr[].");

        StatInit(&stat);
        B.ncol = X.ncol = 0;  // Indicate not to solve the system
        dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);
        
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) {
            printf("LU factorization: dgssvx() returns info %d\n", info);
            if ( info == 0 || info == nrows+1 ) {

                if ( options.PivotGrowth ) printf("Recip. pivot growth = %e\n", rpg);
                if ( options.ConditionNumber ) printf("Recip. condition number = %e\n", rcond);
                Lstore = (SCformat *) L.Store;
                Ustore = (NCformat *) U.Store;
                printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
                printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
                printf("No of nonzeros in L+U = %d\n", Lstore->nnz + Ustore->nnz - nrows);
                printf("FILL ratio = %.1f\n", (float)(Lstore->nnz + Ustore->nnz - nrows)/row_ptr[nrows]);
                printf("L\\U MB %.3f\ttotal MB needed %.3f\n",
                    mem_usage.for_lu/1e6, mem_usage.total_needed/1e6);
                fflush(stdout);
            } else if ( info > 0 && lwork == -1 ) {
                printf("** Estimated memory: %d bytes\n", info - nrows);
            }
        }
        StatFree(&stat);
#elif defined(__x86_64__)
        for (MKL_INT i = 0; i < 64; i++) iparm[i] = 0;
        iparm[0] = 1;// no solver default
        iparm[1] = 2;// Fill-in reordering from METIS
        iparm[3] = 0;// No iterative-direct algorithm
        iparm[4] = 0;// No user fill-in reducing permutation
        iparm[5] = 0;// Write solution into x
        iparm[7] = 0;// Max numbers of iterative refinement steps
        iparm[9] = 13;// Perturb the pivot elements with 1E-13
        iparm[10] = 1;// Use nonsymmetric permutation and scaling MPS
        iparm[12] = 0;// Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy
        iparm[13] = 0;// Output: Number of perturbed pivots
        iparm[17] = -1;// Output: Number of nonzeros in the factor LU
        iparm[18] = -1;// Output: Mflops for LU factorization
        iparm[19] = 0;// Output: Numbers of CG Iterations
        iparm[34] = 1;// PARDISO use C-style indexing for ia and ja arrays */
        maxfct = 1;// Maximum number of numerical factorizations. */
        mnum = 1;// Which factorization to use.
        msglvl = 0;// Print statistical information in file */
        error = 0;// Initialize error flag
        for (MKL_INT i = 0; i < 64; i++) pt[i] = 0;

        phase = 11;
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &nrows, vals, row_ptr, col_idx, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
        if (error != 0 ) {
            printf ("\nERROR during symbolic factorization: %d", error);
            MPI_Abort(MPI_COMM_WORLD, -20230113);
        }

        // 只记录数值分解的时间
        t -= wall_time();
        phase = 22;
        for (int i = 0; i < maxfct; i++)
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &nrows, vals, row_ptr, col_idx, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
        if (error != 0 ) {
            printf ("\nERROR during numerical factorization: %d", error);
            MPI_Abort(MPI_COMM_WORLD, -20230113);
        }
        t += wall_time();
        b_buf = new res_t [nrows];
        t /= maxfct;
#endif
        return t;
    }
    void apply(const res_t * b, res_t * x) {
#ifdef __aarch64__
        const idx_t nrhs = 1;
        options.Fact = FACTORED; // Indicate the factored form of A is supplied.
        X.ncol = B.ncol = nrhs;  // Set the number of right-hand side
        // copy in
        double *rhs = (double*) ((DNformat*) B.Store)->nzval;
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < nrows; i++)
            rhs[i] = b[i];
        StatInit(&stat);
        dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);
        StatFree(&stat);
        // copy back
        double *sol = (double*) ((DNformat*) X.Store)->nzval;
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < nrows; i++)
            x[i] = sol[i];
#elif defined(__x86_64__)
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < nrows; i++)
            b_buf[i] = b[i];
        phase = 33;
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &nrows, vals, row_ptr, col_idx, &idum, &nrhs, iparm, &msglvl, b_buf, x, &error);
        if (error != 0 ) {
            printf ("\nERROR during solution: %d", error);
            MPI_Abort(MPI_COMM_WORLD, -20230113);
        }
#endif
    }
    void fprint_COO(const char * filename) const {
        FILE * fp = fopen(filename, "w+");
        for (idx_t i = 0; i < nrows; i++) {
            idx_t pbeg = row_ptr[i], pend = row_ptr[i+1];
            for (idx_t p = pbeg; p < pend; p++)
                fprintf(fp, "%d %d %.5e\n", i, col_idx[p], vals[p]);
        }
        fclose(fp);
    }
};
#endif

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof=NUM_DOF>
class DenseLU final : public Solver<idx_t, data_t, setup_t, calc_t> {
public:
    DenseLU_type type;
    idx_t num_stencil = 0;// 未初始化
    const idx_t * stencil_offset = nullptr;
    bool setup_called = false;
    double setup_time = 0.0;

    // operator (often as matrix-A)
    const Operator<idx_t, setup_t, setup_t> * oper = nullptr;
    
    idx_t global_dof;
    calc_t * u_data = nullptr, * l_data = nullptr;
    calc_t * dense_x = nullptr, * dense_b = nullptr;// 用于前代回代的数据
    idx_t * sendrecv_cnt = nullptr, * displs = nullptr;
    MPI_Datatype vec_recv_type = MPI_DATATYPE_NULL, mat_recv_type = MPI_DATATYPE_NULL;// 接收者（只有0号进程需要）的数据类型
    MPI_Datatype mat_send_type = MPI_DATATYPE_NULL, vec_send_type = MPI_DATATYPE_NULL;// 发送者（各个进程都需要）的数据类型
#ifndef USE_DENSE
    CSR_sparseMat<idx_t, calc_t, calc_t> * glbA_csr = nullptr;
#endif
    DenseLU(DenseLU_type type) : Solver<idx_t, data_t, setup_t, calc_t>(), type(type) {
        if (type == DenseLU_3D7) {
            num_stencil = 7;
            stencil_offset = stencil_offset_3d7;
        }
        else if (type == DenseLU_3D27) {
            num_stencil = 27;
            stencil_offset = stencil_offset_3d27;
        } 
        else {
            printf("Not supported ilu type %d! Only DenseLU_3d7 or _3d27 available!\n", type);
            MPI_Abort(MPI_COMM_WORLD, -99);
        }
    }
    ~DenseLU() {
        if (u_data != nullptr) {delete u_data; u_data = nullptr;}
        if (l_data != nullptr) {delete l_data; l_data = nullptr;}
        if (dense_b!= nullptr) {delete dense_b; dense_b = nullptr;}
        if (dense_x!= nullptr) {delete dense_x; dense_x = nullptr;}
        if (mat_recv_type != MPI_DATATYPE_NULL) MPI_Type_free(&mat_recv_type);
        if (mat_send_type != MPI_DATATYPE_NULL) MPI_Type_free(&mat_send_type);
        if (vec_recv_type != MPI_DATATYPE_NULL) MPI_Type_free(&vec_recv_type);
        if (vec_send_type != MPI_DATATYPE_NULL) MPI_Type_free(&vec_send_type);
        if (sendrecv_cnt != nullptr) {delete sendrecv_cnt; sendrecv_cnt = nullptr;}
        if (displs != nullptr) {delete displs; displs = nullptr;}
#ifndef USE_DENSE
        if (glbA_csr != nullptr) {delete glbA_csr; glbA_csr = nullptr;}
#endif
    }
    void SetOperator(const Operator<idx_t, setup_t, setup_t> & op) {
        oper = & op;

        this->input_dim[0] = op.input_dim[0];
        this->input_dim[1] = op.input_dim[1];
        this->input_dim[2] = op.input_dim[2];

        this->output_dim[0] = op.output_dim[0];
        this->output_dim[1] = op.output_dim[1];
        this->output_dim[2] = op.output_dim[2];

        Setup();
    }
    void Setup();
    void truncate() { 
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        // if (my_pid == 0) printf("Warning: DenseLU truncated to __fp16\n");
        // idx_t nL = global_dof * (global_dof - 1) / 2;
        // for (idx_t p = 0; p < nL; p++) {
        //     float tmp = (float) l_data[p];
        //     l_data[p] = (data_t) tmp;
        // }
        // idx_t nU = global_dof * (global_dof + 1) / 2;
        // for (idx_t p = 0; p < nU; p++) {
        //     float tmp = (float) u_data[p];
        //     u_data[p] = (data_t) tmp;
        // }
        if (my_pid == 0) printf("Warning: DenseLU NOT Trunc!!!\n");
    }
protected:
    void Mult(const par_structVector<idx_t, calc_t, dof> & input, 
                    par_structVector<idx_t, calc_t, dof> & output) const ;
public:
    void Mult(const par_structVector<idx_t, calc_t, dof> & input, 
                    par_structVector<idx_t, calc_t, dof> & output, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;
        Mult(input, output);
        this->zero_guess = false;// reset for safety concern
    }
};

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof>
void DenseLU<idx_t, data_t, setup_t, calc_t, dof>::Setup()
{
    if (setup_called) return ;
    setup_time = - wall_time();

    assert(this->oper != nullptr);
    // assert matrix has updated halo to prepare data 强制类型转换
    const par_structMatrix<idx_t, setup_t, setup_t, dof> & par_A = *((par_structMatrix<idx_t, setup_t, setup_t, dof>*)(this->oper));
    const seq_structMatrix<idx_t, setup_t, setup_t, dof> & seq_A = *(par_A.local_matrix);// 外层问题的A矩阵
    assert(seq_A.num_diag == num_stencil);

    // 先确定谁来做计算：0号进程来算
    int my_pid;
    MPI_Comm_rank(par_A.comm_pkg->cart_comm, &my_pid);
    int proc_ndim = sizeof(par_A.comm_pkg->cart_ids) / sizeof(int); assert(proc_ndim == 3);
    int num_procs[3], periods[3], coords[3];
    MPI_Cart_get(par_A.comm_pkg->cart_comm, proc_ndim, num_procs, periods, coords);
    // 假定0号进程就是进程网格中的最角落的进程
    if (my_pid == 0) assert(coords[0] == 0 && coords[1] == 0 && coords[2] == 0);

    const idx_t gx = seq_A.local_x * num_procs[1],
                gy = seq_A.local_y * num_procs[0],
                gz = seq_A.local_z * num_procs[2];
    global_dof = gx * gy * gz * dof;// 每个点有dof个自由度
    // const idx_t global_nnz = seq_A.num_diag * global_dof;
    const float sparsity = (float) seq_A.num_diag / (float) global_dof;//global_nnz / (global_dof * global_dof);

    if (global_dof >= 5000 && sparsity < 0.01) {
        if (my_pid == 0) printf("\033[1;32mWARNING! Too large matrix for direct Gauss-Elim method!\033[0m\n");
        // MPI_Abort(MPI_COMM_WORLD, -999);
    }

    dense_x = new calc_t[global_dof];
    dense_b = new calc_t[global_dof];
    setup_t * buf = new setup_t[gx * gy * gz * num_stencil * dof*dof];// 接收缓冲区：全局的稀疏结构化矩阵

    // 执行LU分解，并存储
    idx_t sizes[5], subsizes[5], starts[5];
    // 发送方的矩阵数据类型
    sizes[0] = seq_A.local_y + seq_A.halo_y * 2;    subsizes[0] = seq_A.local_y;    starts[0] = seq_A.halo_y;
    sizes[1] = seq_A.local_x + seq_A.halo_x * 2;    subsizes[1] = seq_A.local_x;    starts[1] = seq_A.halo_x;
    sizes[2] = seq_A.local_z + seq_A.halo_z * 2;    subsizes[2] = seq_A.local_z;    starts[2] = seq_A.halo_z;
    sizes[3] = seq_A.num_diag;                      subsizes[3] = sizes[3];         starts[3] = 0;
    sizes[4] = dof*dof;                             subsizes[4] = sizes[4];         starts[4] = 0;
    MPI_Type_create_subarray(5, sizes, subsizes, starts, MPI_ORDER_C, par_A.comm_pkg->mpi_scalar_type, &mat_send_type);
    MPI_Type_commit(&mat_send_type);
    // 接收方（0号进程）的矩阵数据类型
    MPI_Datatype tmp_type = MPI_DATATYPE_NULL;
    sizes[0] = gy;              subsizes[0] = seq_A.local_y;    starts[0] = 0;
    sizes[1] = gx;              subsizes[1] = seq_A.local_x;    starts[1] = 0;
    sizes[2] = gz;              subsizes[2] = seq_A.local_z;    starts[2] = 0;
    sizes[3] = seq_A.num_diag;  subsizes[3] = sizes[3];         starts[3] = 0;
    sizes[4] = dof*dof;         subsizes[4] = sizes[4];         starts[4] = 0;// starts[]数组改成从头开始，每次都手动指定displs
    MPI_Type_create_subarray(5, sizes, subsizes, starts, MPI_ORDER_C, par_A.comm_pkg->mpi_scalar_type, &tmp_type);
    MPI_Type_create_resized(tmp_type, 0, subsizes[2] * sizes[3] * sizes[4] * sizeof(setup_t), &mat_recv_type);
    MPI_Type_commit(&mat_recv_type);
    // 发送方的向量数据类型
    static_assert(sizeof(calc_t) == 8 || sizeof(calc_t) == 4);
    static_assert(sizeof(setup_t) == 8 || sizeof(setup_t) == 4);
    MPI_Datatype vec_scalar_type = (sizeof(calc_t) == 8) ? MPI_DOUBLE : MPI_FLOAT;
    sizes[0] = seq_A.local_y + seq_A.halo_y * 2;    subsizes[0] = seq_A.local_y;    starts[0] = seq_A.halo_y;
    sizes[1] = seq_A.local_x + seq_A.halo_x * 2;    subsizes[1] = seq_A.local_x;    starts[1] = seq_A.halo_x;
    sizes[2] = seq_A.local_z + seq_A.halo_z * 2;    subsizes[2] = seq_A.local_z;    starts[2] = seq_A.halo_z;
    sizes[3] = dof;                                 subsizes[3] = sizes[3];         starts[3] = 0;
    MPI_Type_create_subarray(4, sizes, subsizes, starts, MPI_ORDER_C, vec_scalar_type, &vec_send_type);
    MPI_Type_commit(&vec_send_type);
    // 接收者（0号进程）的向量数据类型
    sizes[0] = gy;              subsizes[0] = seq_A.local_y;    starts[0] = 0;
    sizes[1] = gx;              subsizes[1] = seq_A.local_x;    starts[1] = 0;
    sizes[2] = gz;              subsizes[2] = seq_A.local_z;    starts[2] = 0;
    sizes[3] = dof;             subsizes[3] = sizes[3];         starts[3] = 0;
    MPI_Type_create_subarray(4, sizes, subsizes, starts, MPI_ORDER_C, vec_scalar_type, &tmp_type);
    MPI_Type_create_resized(tmp_type, 0, subsizes[2] * sizes[3]            * sizeof(calc_t), &vec_recv_type);
    MPI_Type_commit(&vec_recv_type);

    const idx_t tot_procs = num_procs[0] * num_procs[1] * num_procs[2];// py * px * pz
    sendrecv_cnt = new idx_t [tot_procs];
    displs       = new idx_t [tot_procs]; 

    for (idx_t p = 0; p < tot_procs; p++)
        sendrecv_cnt[p] = 1;
    // 这个位移是以resize之后的一个subarray（数组整体）的位移去记的，
    // 刚才已经将extent改为了subsizes[2] * sizes[3] * sizes[4] * sizeof(data_t) 和 subsizes[2] * sizes[3] * sizeof(res_t)
    for (idx_t j = 0; j < num_procs[0]; j++)
        displs[ j * num_procs[1]      * num_procs[2]    ] = j * subsizes[0] * sizes[1] * num_procs[2];// j * stride_y * gx * pz
    for (idx_t j = 0; j < num_procs[0]; j++)
    for (idx_t i = 1; i < num_procs[1]; i++)
        displs[(j * num_procs[1] + i) * num_procs[2]    ] = displs[(j * num_procs[1] + i - 1) * num_procs[2]] + subsizes[1] * num_procs[2];// += stride_x * pz
    for (idx_t j = 0; j < num_procs[0]; j++)
    for (idx_t i = 0; i < num_procs[1]; i++)
    for (idx_t k = 1; k < num_procs[2]; k++)
        displs[(j * num_procs[1] + i) * num_procs[2] + k] = displs[(j * num_procs[1] + i) * num_procs[2] + k - 1] + 1;

    MPI_Allgatherv(seq_A.data, 1, mat_send_type, buf, sendrecv_cnt, displs, mat_recv_type, par_A.comm_pkg->cart_comm);

#ifdef DEBUG
        for (idx_t i = 0; i < global_dof; i++) {
            printf(" idx %4d ", i);
            for (idx_t v = 0; v < seq_A.num_diag; v++)
                printf(" %.6e", buf[i * num_stencil + v]);
            printf("\n");
        }
#endif
    // 上面的AllGatherv还可以优化！！！不要传7*dof*dof=63，减少通信量，压缩至27=(6*dof+dof*dof)
#ifndef USE_DENSE
    // 将结构化排布的稀疏矩阵转成CSR
    idx_t * row_ptr = new idx_t [global_dof+1];
    idx_t * col_idx = new idx_t [global_dof * num_stencil * dof];// 按照最大的可能上限开辟
    calc_t* vals    = new calc_t[global_dof * num_stencil * dof];// + 2是因为对角块不止3个非零元
    int nnz_cnt = 0;
    row_ptr[0] = 0;// init
    for (idx_t j = 0; j < gy; j++)
    for (idx_t i = 0; i < gx; i++)
    for (idx_t k = 0; k < gz; k++)
    for (idx_t f = 0; f < dof; f++) {// 每个结构点上有dof个
        idx_t br = (j * gx + i) * gz + k;
        idx_t row = br * dof + f;
        for (idx_t d = 0; d < num_stencil; d++) {
            idx_t ngb_j = j + stencil_offset[d * 3    ];
            idx_t ngb_i = i + stencil_offset[d * 3 + 1];
            idx_t ngb_k = k + stencil_offset[d * 3 + 2];
            if (ngb_j < 0 || ngb_j >= gy || ngb_i < 0 || ngb_i >= gx || ngb_k < 0 || ngb_k >= gz) continue;
            const setup_t * buf_ptr = buf + (br * num_stencil + d) * dof*dof + f*dof;
            for (idx_t ngb_f = 0; ngb_f < dof; ngb_f++) {
                setup_t v = buf_ptr[ngb_f];
                if (v != 0.0) {
                    idx_t col = ((ngb_j * gx + ngb_i) * gz + ngb_k) * dof + ngb_f;
                    col_idx[nnz_cnt] = col;
                    vals   [nnz_cnt] = v;
                    // printf("%d: j %d i %d k %d f %d ==> d %d\n", 
                    //     nnz_cnt, j,   i,   k,   f,         d);
                    nnz_cnt++;
                }
            }
        }
        row_ptr[row+1] = nnz_cnt;
    }
    glbA_csr = new CSR_sparseMat<idx_t, calc_t, calc_t>(global_dof, row_ptr, col_idx, vals);
    // glbA_csr->fprint_COO("sparseA.txt");
    setup_time += wall_time();
    double decomp_time = glbA_csr->decomp();
    setup_time += decomp_time;
    setup_time -= wall_time();
#else
    setup_t * dense_A = new setup_t[global_dof * global_dof];// 用于分解的稠密A矩阵
    setup_t * L_high = new setup_t [global_dof * (global_dof - 1) / 2];
    setup_t * U_high = new setup_t [global_dof * (global_dof + 1) / 2];
        // 将结构化排布的稀疏矩阵稠密化
        #pragma omp parallel for schedule(static)
        for (idx_t p = 0; p < global_dof * global_dof; p++)
            dense_A[p] = 0.0;
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t j = 0; j < gy; j++)
        for (idx_t i = 0; i < gx; i++)
        for (idx_t k = 0; k < gz; k++)
        for (idx_t d = 0; d < num_stencil; d++) {
            idx_t ngb_j = j + stencil_offset[d * 3 + 0];
            idx_t ngb_i = i + stencil_offset[d * 3 + 1];
            idx_t ngb_k = k + stencil_offset[d * 3 + 2];
            if (ngb_j < 0 || ngb_j >= gy || ngb_i < 0 || ngb_i >= gx || ngb_k < 0 || ngb_k >= gz) continue;
            // 此时该块有dof*dof个元素
            for (idx_t lr = 0; lr < dof; lr++) {
                idx_t start_row = (j * gx + i) * gz + k;
                idx_t gr = start_row * dof + lr;// 全局行序号
                for (idx_t lc = 0; lc < dof; lc++) {
                    idx_t gc = ((ngb_j * gx + ngb_i) * gz + ngb_k) * dof + lc;// 全局列序号
                    dense_A[gr * global_dof + gc] = (calc_t) buf[(start_row * num_stencil + d) * dof*dof + lr*dof + lc];
                }
            }
        }
#ifdef DEBUG
        if (my_pid == 0) {
            char filename[100];
            sprintf(filename, "denseA.solid.txt.%d", my_pid);
            FILE * fp = fopen(filename, "w+");
            for (idx_t i = 0; i < global_dof; i++) {
                for (idx_t j = 0; j < global_dof; j++)
                    fprintf(fp, "%.5e ", dense_A[i * global_dof + j]);
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
#endif
        // 执行分解
        dense_LU_decomp(dense_A, L_high, U_high, global_dof, global_dof);
        delete dense_A;

#ifdef DEBUG
    if (my_pid == 0) {
    // 统计分解完的非零元数目
        idx_t L_nnz = 0, U_nnz = 0, tot_len;
        tot_len = global_dof * (global_dof - 1) / 2;
        for (idx_t i = 0; i < tot_len; i++)
            if (L_high[i] != 0.0) L_nnz ++;
        tot_len = global_dof * (global_dof + 1) / 2;
        for (idx_t i = 0; i < tot_len; i++)
            if (U_high[i] != 0.0) U_nnz ++;
        
        printf(" nnz L %d U %d\n", L_nnz, U_nnz);
        printf("expansion %.2f\n", (float)(L_nnz + U_nnz) / (float)(global_dof * num_stencil) );
    }
#endif

    if constexpr (sizeof(calc_t) == sizeof(setup_t)) {
    // if constexpr (sizeof(calc_t) == sizeof(double)) {
        l_data = L_high;
        u_data = U_high;
    } else {
        if (my_pid == 0) {
            printf("  \033[1;31mWarning\033[0m: LU::Setup() using setup_t of %ld bytes, but calc_t of %ld bytes\n",
                sizeof(setup_t), sizeof(calc_t));
        }
        idx_t tot_len;
        tot_len = global_dof * (global_dof - 1) / 2;
        l_data = new calc_t [tot_len];
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < tot_len; i++)
            l_data[i] = L_high[i];

        tot_len = global_dof * (global_dof + 1) / 2;
        u_data = new calc_t [tot_len];
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < tot_len; i++)
            u_data[i] = U_high[i];
        
        delete L_high;
        delete U_high;
    }
#endif
    delete buf;
    setup_called = true;
    setup_time += wall_time();
}

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof>
void DenseLU<idx_t, data_t, setup_t, calc_t, dof>::Mult(const par_structVector<idx_t, calc_t, dof> & input, par_structVector<idx_t, calc_t, dof> & output) const
{
    CHECK_LOCAL_HALO( *(input.local_vector),  *(output.local_vector));// 检查相容性
    CHECK_OUTPUT_DIM(*this, input);// A * out = in
    CHECK_INPUT_DIM(*this, output);

    int my_pid; MPI_Comm_rank(input.comm_pkg->cart_comm, &my_pid);
    assert(input.global_size_x * input.global_size_y * input.global_size_z * dof == global_dof);

    if (this->zero_guess) {
        const seq_structVector<idx_t, calc_t> & b = *(input.local_vector);
              seq_structVector<idx_t, calc_t> & x = *(output.local_vector);

#ifdef USE_DENSE
        MPI_Allgatherv(b.data, 1, vec_send_type, dense_x, sendrecv_cnt, displs, vec_recv_type, input.comm_pkg->cart_comm);
        dense_forward (l_data, dense_x, dense_b, global_dof, global_dof);// 前代
        dense_backward(u_data, dense_b, dense_x, global_dof, global_dof);// 回代
#else
        MPI_Allgatherv(b.data, 1, vec_send_type, dense_b, sendrecv_cnt, displs, vec_recv_type, input.comm_pkg->cart_comm);
        glbA_csr->apply(dense_b, dense_x);
#endif

        {// 从dense_x中拷回到向量中
            const idx_t ox = output.offset_x     , oy = output.offset_y     , oz = output.offset_z     ;
            const idx_t gx = output.global_size_x,                            gz = output.global_size_z;
            const idx_t lx = x.local_x           , ly = x.local_y           , lz = x.local_z           ;
            const idx_t hx = x.halo_x            , hy = x.halo_y            , hz = x.halo_z            ;
            const idx_t vec_dki_size = x.slice_dki_size, vec_dk_size = x.slice_dk_size;
            for (idx_t j = 0; j < ly; j++)
            for (idx_t i = 0; i < lx; i++)
            for (idx_t k = 0; k < lz; k++) {
                for (idx_t f = 0; f < dof; f++)
                    x.data[(j + hy) * vec_dki_size + (i + hx) * vec_dk_size + (k + hz) * dof + f]
                        = dense_x[(((j + oy) * gx + (i + ox)) * gz + k + oz) * dof + f];
            }
        }

        // if (this->weight != 1.0) vec_mul_by_scalar(this->weight, output, output);
    }
    else {
        assert(false);
        // // 先计算一遍残差
        // par_structVector<idx_t, res_t, dof> resi(input), error(output);
        // this->oper->Mult(output, resi, false);
        // vec_add(input, -1.0, resi, resi);

        // MPI_Allgatherv(resi.local_vector->data, 1, vec_send_type, dense_x, sendrecv_cnt, displs, vec_recv_type, input.comm_pkg->cart_comm);
        // dense_forward (l_data, dense_x, dense_b, global_dof, global_dof);// 前代
        // dense_backward(u_data, dense_b, dense_x, global_dof, global_dof);// 回代

        // {// 从dense_x中拷回到向量中
        //     seq_structVector<idx_t, res_t> & vec = *(error.local_vector);
        //     const idx_t ox = error.offset_x     , oy = error.offset_y, oz = error.offset_z     ;
        //     const idx_t gx = error.global_size_x,                      gz = error.global_size_z;
        //     const idx_t lx = vec.local_x        , ly = vec.local_y   , lz = vec.local_z        ;
        //     const idx_t hx = vec.halo_x         , hy = vec.halo_y    , hz = vec.halo_z         ;
        //     const idx_t vec_dki_size = vec.slice_dki_size, vec_dk_size = vec.slice_dk_size;
        //     for (idx_t j = 0; j < ly; j++)
        //     for (idx_t i = 0; i < lx; i++)
        //     for (idx_t k = 0; k < lz; k++) {
        //         for (idx_t f = 0; f < dof; f++)
        //             vec.data[(j + hy) * vec_dki_size + (i + hx) * vec_dk_size + (k + hz) * dof + f]
        //                 = dense_x[(((j + oy) * gx + (i + ox)) * gz + k + oz) * dof + f];
        //     }
        // }
    
        // vec_add(output, this->weight, error, output);
    }
}


#endif