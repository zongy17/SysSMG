#ifndef SOLID_POINT_GS_HPP
#define SOLID_POINT_GS_HPP

#include "precond.hpp"
#include "../utils/par_struct_mat.hpp"

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof=NUM_DOF>
class PointGS : public Solver<idx_t, data_t, setup_t, calc_t> {
public:
    // 对称GS：0 for sym, 1 for forward, -1 backward
    SCAN_TYPE scan_type = SYMMETRIC;
    mutable bool last_time_forward = false;

    // operator (often as matrix-A)
    const Operator<idx_t, setup_t, setup_t> * oper = nullptr;

    // separate diagonal values if for efficiency concern is needed
    // should only be used when separation is cheap
    bool LUinvD_separated = false;
    seq_structVector<idx_t, data_t, dof*dof> * invD = nullptr;
    seq_structMatrix<idx_t, data_t, calc_t, dof> * L = nullptr, * U = nullptr;
    void separate_LUinvD();

    seq_structVector<idx_t, calc_t, dof> * sqrt_D = nullptr;

    void (*AOS_forward_zero)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*AOS_forward_ALL)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*AOS_backward_zero)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*AOS_backward_ALL)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;

    PointGS() : Solver<idx_t, data_t, setup_t, calc_t>() {  }
    PointGS(SCAN_TYPE type) : Solver<idx_t, data_t, setup_t, calc_t>(), scan_type(type) {
        if (type == FORW_BACK)      last_time_forward = false;
        else if (type == BACK_FORW) last_time_forward = true;
    }

    virtual void SetOperator(const Operator<idx_t, setup_t, setup_t> & op) {
        oper = & op;

        this->input_dim[0] = op.input_dim[0];
        this->input_dim[1] = op.input_dim[1];
        this->input_dim[2] = op.input_dim[2];

        this->output_dim[0] = op.output_dim[0];
        this->output_dim[1] = op.output_dim[1];
        this->output_dim[2] = op.output_dim[2];

        separate_LUinvD();
        const idx_t num_diag = ((const par_structMatrix<idx_t, setup_t, setup_t>&)op).num_diag;
        // const bool scaled = ((const par_structMatrix<idx_t, setup_t, setup_t>&)op).scaled;
        if constexpr (sizeof(calc_t) != sizeof(data_t)) {
            if constexpr (sizeof(calc_t) == 4 && sizeof(data_t) == 2) {// 单-半精度混合计算
                switch (num_diag)
                {
                case 15:
                    AOS_forward_zero = AOS_point_forward_zero_3d_Cal32Stg16<dof, 7>;
                    AOS_forward_ALL  = AOS_point_forward_ALL_3d_Cal32Stg16<dof, 7, 7>;
                    AOS_backward_zero= nullptr;
                    AOS_backward_ALL = AOS_point_backward_ALL_3d_Cal32Stg16<dof, 7, 7>;
                    break;
                case 27:
                    AOS_forward_zero = AOS_point_forward_zero_3d_Cal32Stg16<dof, 13>;
                    AOS_forward_ALL  = AOS_point_forward_ALL_3d_Cal32Stg16<dof, 13, 13>;
                    AOS_backward_zero= nullptr;
                    AOS_backward_ALL = AOS_point_backward_ALL_3d_Cal32Stg16<dof, 13, 13>;
                    break;
                default:
                    MPI_Abort(MPI_COMM_WORLD, -10200);
                }
            }
        }
        else {
            switch (num_diag)
            {
            case 15:
                // AOS_forward_zero = AOS_point_forward_zero_3d15<idx_t, data_t, calc_t, dof>;
                // AOS_forward_ALL  = AOS_point_forward_ALL_3d15<idx_t, data_t, calc_t, dof>;
                // AOS_backward_zero= nullptr;
                // AOS_backward_ALL = AOS_point_backward_ALL_3d15<idx_t, data_t, calc_t, dof>;
                AOS_forward_zero = AOS_point_forward_zero_3d_normal<idx_t, data_t, calc_t, dof, 7>;
                AOS_forward_ALL  = AOS_point_forward_ALL_3d_normal<idx_t, data_t, calc_t, dof, 7, 7>;
                AOS_backward_zero= nullptr;
                AOS_backward_ALL = AOS_point_backward_ALL_3d_normal<idx_t, data_t, calc_t, dof, 7, 7>;
                break;
            case 27:
                AOS_forward_zero = AOS_point_forward_zero_3d_normal<idx_t, data_t, calc_t, dof, 13>;
                AOS_forward_ALL  = AOS_point_forward_ALL_3d_normal<idx_t, data_t, calc_t, dof, 13, 13>;
                AOS_backward_zero= nullptr;
                AOS_backward_ALL = AOS_point_backward_ALL_3d_normal<idx_t, data_t, calc_t, dof, 13, 13>;
                break;
            default:
                MPI_Abort(MPI_COMM_WORLD, -10200);
            }
        }
    }

    virtual void SetScanType(SCAN_TYPE type) {scan_type = type;}

    void truncate() {
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) printf("Warning: PGS truncated invD to __fp16\n");
        assert(LUinvD_separated);
        L->truncate();
        U->truncate();
        const idx_t len = (invD->local_x + invD->halo_x * 2) * (invD->local_y + invD->halo_y *2)
                        * (invD->local_z + invD->halo_z * 2) * dof*dof;
#ifdef __aarch64__
        for (idx_t p = 0; p < len; p++) {
            // float tmp = (float) invD->data[p];
            __fp16 tmp = (__fp16) invD->data[p];
            // if (p == len*dof) printf("PGS::invD truncate %.20e to", invD->data[p]);
            invD->data[p] = (data_t) tmp;
            // if (p == len*dof) printf("%.20e\n", invD->data[p]);
        }
#else
        printf("architecture not support truncated to fp16\n");
#endif
    }

protected:
    // 近似求解一个（残差）方程，以b为右端向量，返回x为近似解
    void Mult(const par_structVector<idx_t, calc_t, dof> & b, 
                    par_structVector<idx_t, calc_t, dof> & x) const;
public:
    void Mult(const par_structVector<idx_t, calc_t, dof> & b,
                    par_structVector<idx_t, calc_t, dof> & x, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;
        Mult(b, x);
        this->zero_guess = false;// reset for safety concern
    }

    void ForwardPass(const par_structVector<idx_t, calc_t, dof> & b, par_structVector<idx_t, calc_t, dof> & x) const;
    void BackwardPass(const par_structVector<idx_t, calc_t, dof> & b, par_structVector<idx_t, calc_t, dof> & x) const;

    virtual ~PointGS();
};

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof>
PointGS<idx_t, data_t, setup_t, calc_t, dof>::~PointGS() {
    if (LUinvD_separated) {
        if (invD != nullptr) {delete invD; invD = nullptr;}
        if (L != nullptr) {delete L; L = nullptr;}
        if (U != nullptr) {delete U; U = nullptr;}
    }
}

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof>
void PointGS<idx_t, data_t, setup_t, calc_t, dof>::Mult(const   par_structVector<idx_t, calc_t, dof> & b, 
                                                                par_structVector<idx_t, calc_t, dof> & x) const {
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);

#ifdef PROFILE
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    int num_procs; MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    double t = 0.0, bytes, mint, maxt;
#endif

        switch (scan_type)
        {
        // 零初值优化，注意在对称GS中的后扫和单纯后扫的零初值优化是不一样的，小心！！！
        // 需要注意这个x传进来时可能halo区并不是0，如果不设置0并通信更新halo区，会导致使用错误的值，从而收敛性变慢
        // 还需要注意，这个x传进来时可能数据区不是0，如果不采用0初值的优化代码，而直接采用naive版本但是又没有数据区清零，则会用到旧值，从而收敛性变慢
        case SYMMETRIC:
            if (this->zero_guess) 
                x.set_val(0.0, true);
            else
                x.update_halo();
#ifdef PROFILE
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time();
#endif
            ForwardPass(b, x);
#ifdef PROFILE
            t = wall_time() - t;
            MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, b.comm_pkg->cart_comm);
            MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, b.comm_pkg->cart_comm);
            if (my_pid == 0) {
                int num_diag = L->num_diag + 1 + (this->zero_guess ? 0 : U->num_diag);
                int num_vec = sqrt_D ? 2 : 3;
                bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
                      * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(calc_t) * num_vec * NUM_DOF;// 向量的数据量
                bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z
                        * num_diag * NUM_DOF*NUM_DOF* sizeof(data_t);
                bytes *= num_procs;
                bytes /= (1024 * 1024 * 1024);// GB
                printf("PGS-F data %ld calc %ld d%dv%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                     sizeof(data_t), sizeof(calc_t), num_diag, num_vec, bytes, mint, maxt, bytes/maxt, bytes/mint);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time();
#endif
            // 是否要注释掉这行决定后扫零初值的，取决于迭代次数是否会被显著影响
            // 一般在前扫和后扫中间加一次通信，有利于减少迭代数，次数和访存量的减少需要权衡
            this->zero_guess = false;
            x.update_halo();// 通信完之后halo区是非零的，用普通版本的后扫
#ifdef PROFILE
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time();
#endif
            BackwardPass(b, x);
#ifdef PROFILE
            t = wall_time() - t;
            MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, b.comm_pkg->cart_comm);
            MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, b.comm_pkg->cart_comm);
            if (my_pid == 0) {
                int num_diag = L->num_diag + 1 + (this->zero_guess ? 0 : U->num_diag);
                int num_vec = sqrt_D ? 2 : 3;
                bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
                      * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(calc_t) * num_vec * NUM_DOF;// 向量的数据量
                bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z
                        * num_diag * NUM_DOF*NUM_DOF* sizeof(data_t);
                bytes *= num_procs;
                bytes /= (1024 * 1024 * 1024);// GB
                printf("PGS-B data %ld calc %ld d%dv%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                     sizeof(data_t), sizeof(calc_t), num_diag, num_vec, bytes, mint, maxt, bytes/maxt, bytes/mint);
            }
#endif
            break;
        case FORWARD:
            if (this->zero_guess)
                x.set_val(0.0, true);
            else
                x.update_halo();
            ForwardPass(b, x);
            break;
        case BACKWARD:
            if (this->zero_guess)
                x.set_val(0.0, true);
            else
                x.update_halo();
            BackwardPass(b, x);
            break; 
        case FORW_BACK:
        case BACK_FORW:
            if (this->zero_guess)
                x.set_val(0.0, true);
            else
                x.update_halo();
            if (last_time_forward) {
                BackwardPass(b, x);
                last_time_forward = false;
            } else {
                ForwardPass(b, x);
                last_time_forward = false;
            }
            break;
        default:// do nothing, just act as an identity operator
            vec_copy(b, x);
            break;
        }
}

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof>
void PointGS<idx_t, data_t, setup_t, calc_t, dof>::ForwardPass(const    par_structVector<idx_t, calc_t, dof> & b, 
                                                                        par_structVector<idx_t, calc_t, dof> & x) const
{
    const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
    assert(LUinvD_separated);
    CHECK_LOCAL_HALO(x_vec, b_vec);
    CHECK_LOCAL_HALO(x_vec, *L);

    const idx_t num_diag = (L->num_diag << 1) + 1;

    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;
    const data_t * L_data = L->data, * U_data = U->data, * invD_data = invD->data;
    const calc_t * sqD_data = sqrt_D ? sqrt_D->data : nullptr;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_dk_size = x_vec.slice_dk_size, vec_dki_size = x_vec.slice_dki_size;
    const idx_t mat_edki_size = L->slice_edki_size, mat_edk_size = L->slice_edk_size, mat_ed_size = L->slice_ed_size;
    const idx_t invD_dk_size = invD->slice_dk_size, invD_dki_size = invD->slice_dki_size;
    const idx_t elms = dof*dof;

    const calc_t weight = this->weight;
    const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const calc_t,
        const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    if (this->zero_guess) {
        kernel = sqrt_D ? nullptr : AOS_forward_zero; 
    } else {
        kernel = sqrt_D ? nullptr : AOS_forward_ALL;
    }
    assert(kernel);

    if (num_threads > 1) {
        const idx_t slope = (num_diag == 7 || num_diag == 15) ? 1 : 2;
        idx_t dim_0 = jend - jbeg, dim_1 = iend - ibeg;
        idx_t flag[dim_0 + 1];
        flag[0] = dim_1 - 1;// 边界标记已完成
        for (idx_t j = 0; j < dim_0; j++) 
            flag[j + 1] = -1;// 初始化为-1
        const idx_t wait_offi = slope - 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            // 各自开始计算
            idx_t nlevs = dim_1 + (dim_0 - 1) * slope;
            for (idx_t lid = 0; lid < nlevs; lid++) {
                // 每层的起始点位于左上角
                idx_t jstart_lev = MIN(lid / slope, dim_0 - 1);
                idx_t istart_lev = lid - slope * jstart_lev;
                idx_t ntasks = MIN(jstart_lev + 1, ((dim_1-1) - istart_lev) / slope + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                idx_t t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                idx_t t_end = t_beg + my_cnt;

                for (idx_t it = t_end - 1; it >= t_beg; it--) {
                    idx_t j_lev = jstart_lev - it;
					idx_t i_lev = istart_lev + it * slope;
                    idx_t j = jbeg + j_lev, i = ibeg + i_lev;// 用于数组选址计算的下标
                    idx_t i_to_wait = (i == iend - 1) ? i_lev : (i_lev + wait_offi);
                    const idx_t vec_off = j * vec_dki_size  + i * vec_dk_size  + kbeg * dof;
                    const idx_t mat_off = j * mat_edki_size + i * mat_edk_size + kbeg * mat_ed_size;
                    const idx_t invD_off= j * invD_dki_size + i * invD_dk_size + kbeg * elms;
                    const data_t * L_jik = L_data + mat_off, * U_jik = U_data + mat_off, * invD_jik = invD_data + invD_off;
                    const calc_t * sqD_jik = sqD_data ? (sqD_data + vec_off) : nullptr;
                    calc_t * x_jik = x_data + vec_off;
                    const calc_t * b_jik = b_data + vec_off;
                    // 线程边界处等待
                    if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) < i_lev - 1) {  } // 只需检查W依赖
                    if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) < i_to_wait) {  }
                    
                    // 中间的不需等待
                    kernel(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik);

                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev+1], i_lev, __ATOMIC_RELEASE);
                    else flag[j_lev+1] = i_lev;
                }
            }
        }
    }
    else {
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            const idx_t vec_off = j * vec_dki_size  + i * vec_dk_size  + kbeg * dof;
            const idx_t mat_off = j * mat_edki_size + i * mat_edk_size + kbeg * mat_ed_size;
            const idx_t invD_off= j * invD_dki_size + i * invD_dk_size + kbeg * elms;
            const data_t * L_jik = L_data + mat_off, * U_jik = U_data + mat_off, * invD_jik = invD_data + invD_off;
            const calc_t * sqD_jik = sqD_data ? (sqD_data + vec_off) : nullptr;
            calc_t * x_jik = x_data + vec_off;
            const calc_t * b_jik = b_data + vec_off;
            // printf("j %d i %d\n", j, i);
            kernel(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik);
        }
    }
}

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof>
void PointGS<idx_t, data_t, setup_t, calc_t, dof>::BackwardPass(const   par_structVector<idx_t, calc_t, dof> & b, 
                                                                        par_structVector<idx_t, calc_t, dof> & x) const
{
    const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
    assert(LUinvD_separated);
    CHECK_LOCAL_HALO(x_vec, b_vec);
    CHECK_LOCAL_HALO(x_vec, *U);

    const idx_t num_diag = (U->num_diag << 1) + 1;

    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;
    const data_t * L_data = L->data, * U_data = U->data, * invD_data = invD->data;
    const calc_t * sqD_data = sqrt_D ? sqrt_D->data : nullptr;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_dk_size = x_vec.slice_dk_size, vec_dki_size = x_vec.slice_dki_size;
    const idx_t mat_edki_size = L->slice_edki_size, mat_edk_size = L->slice_edk_size, mat_ed_size = L->slice_ed_size;
    const idx_t invD_dk_size = invD->slice_dk_size, invD_dki_size = invD->slice_dki_size;
    const idx_t elms = dof*dof;

    const calc_t weight = this->weight;
    const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const calc_t,
        const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    if (this->zero_guess) {
        kernel = sqrt_D ? nullptr : AOS_backward_zero; 
    } else {
        kernel = sqrt_D ? nullptr : AOS_backward_ALL;
    }
    assert(kernel);

    if (num_threads > 1) {// level-based的多线程并行
        const idx_t slope = (num_diag == 7 || num_diag == 15) ? 1 : 2;
        idx_t dim_0 = jend - jbeg, dim_1 = iend - ibeg;
        idx_t flag[dim_0 + 1];
        flag[dim_0] = 0;// 边界标记已完成
        for (idx_t j = 0; j < dim_0; j++) 
            flag[j] = dim_1;// 初始化
        const idx_t wait_offi = - (slope - 1);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            // 各自开始计算
            idx_t nlevs = dim_1 + (dim_0 - 1) * slope;
            for (idx_t lid = nlevs - 1; lid >= 0; lid--) {
                // 每层的起始点位于左上角
                idx_t jstart_lev = MIN(lid / slope, dim_0 - 1);
                idx_t istart_lev = lid - slope * jstart_lev;
                idx_t ntasks = MIN(jstart_lev + 1, ((dim_1-1) - istart_lev) / slope + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                idx_t t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                idx_t t_end = t_beg + my_cnt;

                for (idx_t it = t_beg; it < t_end; it++) {
                    idx_t j_lev = jstart_lev - it;
					idx_t i_lev = istart_lev + it * slope;
                    idx_t j = jbeg + j_lev, i = ibeg + i_lev;// 用于数组选址计算的下标
                    idx_t i_to_wait = (i == ibeg) ? i_lev : (i_lev + wait_offi);
                    const idx_t vec_off = j * vec_dki_size  + i * vec_dk_size  + kend * dof;
                    const idx_t mat_off = j * mat_edki_size + i * mat_edk_size + kend * mat_ed_size;
                    const idx_t invD_off= j * invD_dki_size + i * invD_dk_size + kend * elms;
                    const data_t * L_jik = L_data + mat_off, * U_jik = U_data + mat_off, * invD_jik = invD_data + invD_off;
                    const calc_t * sqD_jik = sqD_data ? (sqD_data + vec_off) : nullptr;
                    calc_t * x_jik = x_data + vec_off;
                    const calc_t * b_jik = b_data + vec_off;
                    // 线程边界处等待
                    if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) > i_to_wait) {  }
                    if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) > i_lev + 1) {  }
                    // 中间的不需等待
                    kernel(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik);
                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev], i_lev, __ATOMIC_RELEASE);
                    else flag[j_lev] = i_lev;
                }
            }
        }
    }
    else {
        for (idx_t j = jend - 1; j >= jbeg; j--)
        for (idx_t i = iend - 1; i >= ibeg; i--) {
            const idx_t vec_off = j * vec_dki_size  + i * vec_dk_size  + kend * dof;
            const idx_t mat_off = j * mat_edki_size + i * mat_edk_size + kend * mat_ed_size;
            const idx_t invD_off= j * invD_dki_size + i * invD_dk_size + kend * elms;
            const data_t * L_jik = L_data + mat_off, * U_jik = U_data + mat_off, * invD_jik = invD_data + invD_off;
            const calc_t * sqD_jik = sqD_data ? (sqD_data + vec_off) : nullptr;
            calc_t * x_jik = x_data + vec_off;
            const calc_t * b_jik = b_data + vec_off;
            kernel(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik);
        }
    }
}

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof>
void PointGS<idx_t, data_t, setup_t, calc_t, dof>::separate_LUinvD() {
    assert(this->oper != nullptr);
    assert(!LUinvD_separated);
    const par_structMatrix<idx_t, setup_t, setup_t, dof> & par_A = *((par_structMatrix<idx_t, setup_t, setup_t, dof>*)this->oper);
    const seq_structMatrix<idx_t, setup_t, setup_t, dof> & seq_A  = *(par_A.local_matrix);
    idx_t diag_id = seq_A.num_diag >> 1;// 7 >> 1 = 3, 15 >> 1 = 7, 27 >> 1 = 13
    
    const idx_t hx = seq_A.halo_x , hy = seq_A.halo_y , hz = seq_A.halo_z ;
    const idx_t lx = seq_A.local_x, ly = seq_A.local_y, lz = seq_A.local_z;
    assert(dof*dof == seq_A.elem_size);
    invD = new seq_structVector<idx_t, data_t, dof*dof>(lx, ly, lz, hx, hy, hz);
    L = new seq_structMatrix<idx_t, data_t, calc_t, dof>(diag_id, lx, ly, lz, hx, hy, hz);
    U = new seq_structMatrix<idx_t, data_t, calc_t, dof>(diag_id, lx, ly, lz, hx, hy, hz);

    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if constexpr (sizeof(setup_t) != sizeof(data_t)) {
        if (my_pid == 0) 
            printf(" WARNING: PGS Setop truncate setup_t of %ld to data_t of %ld bytes\n", sizeof(setup_t), sizeof(data_t));
    }

    const idx_t jbeg = hy, jend = jbeg + ly,// 注意这里经常写bug
                ibeg = hx, iend = ibeg + lx,
                kbeg = hz, kend = kbeg + lz;
    #pragma omp parallel 
    {
        setup_t buf[dof * dof], inv_res[dof * dof];
        #pragma omp for collapse(3) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++)
        for (idx_t k = kbeg; k < kend; k++) {
            const setup_t * src_ptr = seq_A.data + j * seq_A.slice_edki_size + i * seq_A.slice_edk_size + k * seq_A.slice_ed_size;
                    data_t *  L_ptr = L->data    + j *    L->slice_edki_size + i *    L->slice_edk_size + k *    L->slice_ed_size;
                    data_t *  U_ptr = U->data    + j *    U->slice_edki_size + i *    U->slice_edk_size + k *    U->slice_ed_size;
                data_t   * invD_ptr = invD->data + j * invD->slice_dki_size  + i * invD->slice_dk_size  + k * dof*dof;
            const idx_t copy_nelems = diag_id * seq_A.elem_size;
            for (idx_t p = 0; p < copy_nelems; p++)// L部分
                L_ptr[p] = src_ptr[p];
            src_ptr += copy_nelems;
            
            memcpy(buf, src_ptr, sizeof(setup_t) * dof * dof);
            matinv_row<idx_t, setup_t, dof>(buf, inv_res);
            for (idx_t p = 0; p < dof*dof; p++)
                invD_ptr[p] = inv_res[p];
            // for (idx_t f = 0; f < seq_A.elem_size; f++) printf("%.5e ", invD_ptr[f]);
            // printf("\n");

            src_ptr += dof*dof;
            for (idx_t p = 0; p < copy_nelems; p++)// U部分
                U_ptr[p] = src_ptr[p];
        }
    }

    if (par_A.scaled) {
        const seq_structVector<idx_t, setup_t, dof> & src_h = *(par_A.sqrt_D);
        sqrt_D = new seq_structVector<idx_t, calc_t, dof>(lx, ly, lz, hx, hy, hz);
        const idx_t tot_elems = (lx + hx*2) * (ly + hy*2) * (lz + hz*2) * dof;
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < tot_elems; i++)
            sqrt_D->data[i] = src_h.data[i];
    }

    LUinvD_separated = true;
}

#endif