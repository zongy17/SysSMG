#ifndef SOLID_POINT_GS_HPP
#define SOLID_POINT_GS_HPP

#include "precond.hpp"
#include "../utils/par_struct_mat.hpp"

template<typename idx_t, typename data_t, typename setup_t, int dof=NUM_DOF>
struct IrrgPts_Effect
{
    idx_t loc_id;// local idx to access irrgPts_vec array
    idx_t i, j, k;// 该非规则点所影响的结构点的三维全局坐标
    data_t val[dof*dof];// 影响的值
    IrrgPts_Effect() {}
    IrrgPts_Effect(idx_t ir, idx_t i, idx_t j, idx_t k, const setup_t* v)
        : loc_id(ir), i(i), j(j), k(k) {
        for (idx_t f = 0; f < dof*dof; f++)
            val[f] = v[f];// deep copy
    }
    bool operator < (const IrrgPts_Effect & b) const {
        if (j < b.j) return true;
        else if (j > b.j) return false;
        else {// j == b.j
            if (i < b.i) return true;
            else if (i > b.i) return false;
            else {// i == b.i
                assert(k != b.k);
                return k < b.k;
            }
        }
    }
};

template<typename idx_t, typename data_t, int dof=NUM_DOF>
struct IrrgPts_InvMat
{
    idx_t gid;// 非规则点的全局索引
    data_t val[dof*dof];
};

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
    seq_structMatrix<idx_t, data_t, calc_t, 1> * L_cprs = nullptr, * U_cprs = nullptr;
    void separate_LUinvD();

    seq_structVector<idx_t, calc_t, dof> * sqrt_D = nullptr;

    idx_t num_irrgPts_invD = 0;
    IrrgPts_InvMat<idx_t, data_t, dof> * irrgPts_invD = nullptr;
    // 将A矩阵内的非规则点对结构点的影响提取出来，并按照自然序排序，以便在做结构点
    void prepare_irrgPts();
    idx_t num_irrgPts_effect = 0;
    IrrgPts_Effect<idx_t, data_t, setup_t, dof> * irrg_to_Struct = nullptr;// 局部非规则点的一维序号，对应结构点的三维坐标（已带偏移），本结构点受非结构点的影响

    void (*AOS_forward_zero)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*AOS_forward_ALL)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*AOS_backward_zero)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*AOS_backward_ALL)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    
    void (*AOS_forward_zero_irr)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*,
        const idx_t /* which k */, const calc_t* /* contrib to this k */) = nullptr;
    void (*AOS_forward_ALL_irr)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*,
        const idx_t /* which k */, const calc_t* /* contrib to this k */) = nullptr;
    void (*AOS_backward_zero_irr)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*,
        const idx_t /* which k */, const calc_t* /* contrib to this k */) = nullptr;
    void (*AOS_backward_ALL_irr)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*,
        const idx_t /* which k */, const calc_t* /* contrib to this k */) = nullptr;

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
        prepare_irrgPts();
        const idx_t num_diag = ((const par_structMatrix<idx_t, setup_t, setup_t>&)op).num_diag;
        if constexpr (sizeof(calc_t) == 4 && sizeof(data_t) == 2) {// 单-半精度混合计算
            switch (num_diag)
            {
            case  7:
                    AOS_forward_zero = sqrt_D ? nullptr
                                        :   AOS_point_forward_zero_3d_Cal32Stg16<dof, 3>;
                    AOS_forward_ALL  = sqrt_D ? nullptr
                                        :   AOS_point_forward_ALL_3d_Cal32Stg16<dof, 3, 3>;
                    AOS_backward_zero= nullptr;
                    AOS_backward_ALL = sqrt_D ? nullptr
                                        :   AOS_point_backward_ALL_3d_Cal32Stg16<dof, 3, 3>;

                    AOS_forward_zero_irr = sqrt_D ? nullptr
                                            :   AOS_point_forward_zero_3d_irr_Cal32Stg16<dof, 3>;
                    AOS_forward_ALL_irr  = sqrt_D ? nullptr
                                            :   AOS_point_forward_ALL_3d_irr_Cal32Stg16<dof, 3, 3>;
                    AOS_backward_zero_irr= nullptr;
                    AOS_backward_ALL_irr = sqrt_D ? nullptr
                                            :   AOS_point_backward_ALL_3d_irr_Cal32Stg16<dof, 3, 3>;
                break;
            default:
                MPI_Abort(MPI_COMM_WORLD, -10200);
            }
        }
        else {
            switch (num_diag)
            {
            case  7:
                    AOS_forward_zero = sqrt_D ? AOS_point_forward_zero_3d_scaled_normal<idx_t, data_t, calc_t, dof, 3>
                                            :   AOS_point_forward_zero_3d_normal<idx_t, data_t, calc_t, dof, 3>;
                    AOS_forward_ALL  = sqrt_D ? AOS_point_forward_ALL_3d_scaled_normal<idx_t, data_t, calc_t, dof, 3, 3>
                                            :   AOS_point_forward_ALL_3d_normal<idx_t, data_t, calc_t, dof, 3, 3>;
                    AOS_backward_zero= nullptr;
                    AOS_backward_ALL = sqrt_D ? AOS_point_backward_ALL_3d_scaled_normal<idx_t, data_t, calc_t, dof, 3, 3>
                                            :   AOS_point_backward_ALL_3d_normal<idx_t, data_t, calc_t, dof, 3, 3>;
                    AOS_forward_zero_irr = sqrt_D ? nullptr
                                            :   AOS_point_forward_zero_3d_normal_irr<idx_t, data_t, calc_t, dof, 3>;
                    AOS_forward_ALL_irr  = sqrt_D ? nullptr
                                            :   AOS_point_forward_ALL_3d_normal_irr<idx_t, data_t, calc_t, dof, 3, 3>;
                    AOS_backward_zero_irr= nullptr;
                    AOS_backward_ALL_irr = sqrt_D ? nullptr
                                            :   AOS_point_backward_ALL_3d_normal_irr<idx_t, data_t, calc_t, dof, 3, 3>;
                break;
            default:
                MPI_Abort(MPI_COMM_WORLD, -10203);
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
        for (idx_t ir = 0; ir < num_irrgPts_invD; ir++) {
            for (idx_t f = 0; f < dof*dof; f++) {
                __fp16 tmp = (__fp16) irrgPts_invD[ir].val[f];
                irrgPts_invD[ir].val[f] = (data_t) tmp;
            }
        }
        // 非规则点
        for (idx_t ir = 0; ir < num_irrgPts_effect; ir++) {
            for (idx_t f = 0; f < dof*dof; f++) {
                __fp16 tmp = (__fp16) irrg_to_Struct[ir].val[f];
                irrg_to_Struct[ir].val[f] = (data_t) tmp;
            }
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
    if (num_irrgPts_invD > 0) {
        delete irrgPts_invD; irrgPts_invD = nullptr;
    }
    if (num_irrgPts_effect > 0) {
        delete irrg_to_Struct; irrg_to_Struct = nullptr;
    }
}

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof>
void PointGS<idx_t, data_t, setup_t, calc_t, dof>::prepare_irrgPts()
{
    const par_structMatrix<idx_t, setup_t, setup_t, dof> & par_A = *((par_structMatrix<idx_t, setup_t, setup_t, dof> *)(this->oper));
    if (num_irrgPts_effect != 0)
        delete irrg_to_Struct;
    
    std::vector<IrrgPts_Effect<idx_t, data_t, setup_t, dof> > container;
    for (idx_t ir = 0; ir < par_A.num_irrgPts; ir++) {
        idx_t pbeg = par_A.irrgPts[ir].beg, pend = pbeg + par_A.irrgPts[ir].nnz;
        for (idx_t p = pbeg; p < pend; p++) {
            if (par_A.irrgPts_ngb_ijk[p*3] != -1) {
                idx_t loc_i = par_A.irrgPts_ngb_ijk[p*3  ] - par_A.offset_x + par_A.local_matrix->halo_x;
                idx_t loc_j = par_A.irrgPts_ngb_ijk[p*3+1] - par_A.offset_y + par_A.local_matrix->halo_y;
                idx_t loc_k = par_A.irrgPts_ngb_ijk[p*3+2] - par_A.offset_z + par_A.local_matrix->halo_z;
                const setup_t * val = par_A.irrgPts_A_vals + (p*2+1) *dof*dof;// 我（非规则点）对别人结构点的影响
                IrrgPts_Effect<idx_t, data_t, setup_t, dof> obj(ir, loc_i, loc_j, loc_k, val);
                container.push_back(obj);
            }
        }
    }
    std::sort(container.begin(), container.end());
    // check sorted
    num_irrgPts_effect = container.size();
    for (idx_t i = 0; i < num_irrgPts_effect - 1; i++) {
        idx_t curr_i = container[i].i;
        idx_t curr_j = container[i].j;
        idx_t curr_k = container[i].k;
        idx_t next_i = container[i+1].i;
        idx_t next_j = container[i+1].j;
        idx_t next_k = container[i+1].k;
#ifdef DISABLE_OMP
        assert(curr_j < next_j || (curr_j==next_j && curr_i < next_i) || (curr_j==next_j && curr_i==next_i && curr_k < next_k));
#else
        // 取巧的写法：因为特别地，在做level-based的并行sptrsv时，
        // 每个level内都只有一个点会被非规则点影响，且这些点的(x,z)坐标都相同
        assert(curr_i == next_i && curr_k == next_k);
        assert(curr_j < next_j);
#endif
    }
    // Copy
    irrg_to_Struct = new IrrgPts_Effect<idx_t, data_t, setup_t, dof> [num_irrgPts_effect];
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    for (idx_t i = 0; i < num_irrgPts_effect; i++) {
        irrg_to_Struct[i] = container[i];
#ifdef DEBUG
        printf(" proc %d locally %d => (%d,%d,%d) of ", 
            my_pid, irrg_to_Struct[i].loc_id, irrg_to_Struct[i].i, irrg_to_Struct[i].j, irrg_to_Struct[i].k);
        for (idx_t f = 0; f < dof*dof; f++)
            printf(" %.6e", irrg_to_Struct[i].val[f]);
#endif
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
                int num;
                if (L_cprs) {
                    num = dof*dof + L_cprs->num_diag;
                    if (this->zero_guess == false) num += U_cprs->num_diag; 
                } else {
                    num = dof*dof * (L->num_diag + 1);
                    if (this->zero_guess == false) num += dof*dof * U->num_diag;
                }
                int num_vec = sqrt_D ? 2 : 3;
                bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
                      * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(calc_t) * num_vec * dof;// 向量的数据量
                bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num * sizeof(data_t);
                bytes *= num_procs;
                bytes /= (1024 * 1024 * 1024);// GB
                printf("PGS-F data %ld calc %ld B%dv%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                     sizeof(data_t), sizeof(calc_t), num, num_vec, bytes, mint, maxt, bytes/maxt, bytes/mint);
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
                int num;
                if (U_cprs) {
                    num = dof*dof + U_cprs->num_diag;
                    if (this->zero_guess == false) num += L_cprs->num_diag; 
                } else {
                    num = dof*dof * (U->num_diag + 1);
                    if (this->zero_guess == false) num += dof*dof * L->num_diag;
                }
                int num_vec = sqrt_D ? 2 : 3;
                bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
                      * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(calc_t) * num_vec * dof;// 向量的数据量
                bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num * sizeof(data_t);
                bytes *= num_procs;
                bytes /= (1024 * 1024 * 1024);// GB
                printf("PGS-B data %ld calc %ld B%dv%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                     sizeof(data_t), sizeof(calc_t), num, num_vec, bytes, mint, maxt, bytes/maxt, bytes/mint);
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
    const par_structMatrix<idx_t, setup_t, setup_t, dof> * par_A = (par_structMatrix<idx_t, setup_t, setup_t, dof>*)(this->oper);
    assert(LUinvD_separated);
    CHECK_LOCAL_HALO(x_vec, b_vec);

    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;
    idx_t num_diag;
    const data_t * L_data = nullptr, * U_data = nullptr;
    idx_t mat_edki_size, mat_edk_size, mat_ed_size;
    if (L_cprs) {
        CHECK_LOCAL_HALO(x_vec, *L_cprs);
        assert(U_cprs);
        num_diag = L_cprs->num_diag / dof;
        L_data = L_cprs->data; U_data = U_cprs->data;
        mat_edki_size = L_cprs->slice_edki_size;
        mat_edk_size  = L_cprs->slice_edk_size ;
        mat_ed_size   = L_cprs->slice_ed_size  ;
    } else {
        CHECK_LOCAL_HALO(x_vec, *L);
        num_diag = (L->num_diag << 1) + 1;
        L_data = L->data; U_data = U->data;
        mat_edki_size = L->slice_edki_size;
        mat_edk_size  = L->slice_edk_size ;
        mat_ed_size   = L->slice_ed_size  ;
    }
    const data_t * invD_data = invD->data;
    const calc_t * sqD_data = sqrt_D ? sqrt_D->data : nullptr;
    const idx_t vec_dk_size = x_vec.slice_dk_size, vec_dki_size = x_vec.slice_dki_size;
    const idx_t invD_dk_size = invD->slice_dk_size, invD_dki_size = invD->slice_dki_size;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t elms = dof*dof;

    const calc_t weight = this->weight;
    const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const calc_t,
        const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*kernel_irr) (const idx_t, const idx_t, const idx_t, const calc_t,
        const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*, const idx_t, const calc_t*) = nullptr;
    kernel = this->zero_guess ? AOS_forward_zero : AOS_forward_ALL;
    kernel_irr = this->zero_guess ? AOS_forward_zero_irr : AOS_forward_ALL_irr;

    assert(kernel);
    assert(kernel_irr);

    // int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    // printf("proc %d A.num_irr %d\n", my_pid, par_A->num_irrgPts);

    // 前扫先处理非规则点
    assert(par_A->num_irrgPts == x.num_irrgPts && x.num_irrgPts == b.num_irrgPts);
    for (idx_t ir = 0; ir < par_A->num_irrgPts; ir++) {
        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid && x.irrgPts[ir].gid == b.irrgPts[ir].gid);
        assert(par_A->irrgPts[ir].gid == irrgPts_invD[ir].gid);
        const idx_t pbeg = par_A->irrgPts[ir].beg, pend = pbeg + par_A->irrgPts[ir].nnz;
        calc_t tmp[dof], tmp2[dof];
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)// tmp = -b
            tmp[f] = - b.irrgPts[ir].val[f];
        assert(par_A->irrgPts_ngb_ijk[(pend-1)*3] == -1);// 对角元位置
        data_t * invD_ptr = irrgPts_invD[ir].val;
        #pragma omp parallel for schedule(static) reduction(+:tmp)// 编译期定长数组有效
        for (idx_t p = pbeg; p < pend - 1; p++) {// 跳过了对角元
            const idx_t ngb_i = par_A->irrgPts_ngb_ijk[p*3  ],
                        ngb_j = par_A->irrgPts_ngb_ijk[p*3+1],
                        ngb_k = par_A->irrgPts_ngb_ijk[p*3+2];// global coord
            const idx_t i = ibeg + ngb_i - par_A->offset_x,
                        j = jbeg + ngb_j - par_A->offset_y,
                        k = kbeg + ngb_k - par_A->offset_z;
            const setup_t * src_ptr = par_A->irrgPts_A_vals + (p<<1)*elms;
            data_t dst_ptr[dof*dof];
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof*dof; f++)
                dst_ptr[f] = src_ptr[f];
            matvec_mla<idx_t, data_t, calc_t, dof>(dst_ptr, x_data + j*vec_dki_size + i*vec_dk_size +  k*dof, tmp);
        }// 此时 tmp = - b + L*x^{t} + U*x^{t}
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_ptr, tmp, tmp2, - weight);// tmp2 = w*D^{-1}*(b - L*x^{t} - U*x^{t})
        #pragma GCC unroll 4
        for (idx_t f = 0; f < dof; f++)
            x.irrgPts[ir].val[f] = (1.0 - weight) * x.irrgPts[ir].val[f] + tmp2[f];
    }
    // 再处理结构点：边遍历三维向量边检查是否碰到非规则的邻居
    idx_t ptr = 0, irr_ngb_i = -1, irr_ngb_j = -1, irr_ngb_k = -1;
    bool need_to_check = ptr < num_irrgPts_effect;
    // printf(" proc %d got %d need_to %d\n", my_pid, num_irrgPts_effect, need_to_check);
    if (need_to_check) {
        irr_ngb_i = irrg_to_Struct[ptr].i;  irr_ngb_j = irrg_to_Struct[ptr].j;  irr_ngb_k = irrg_to_Struct[ptr].k;
        // printf(" proc %d before : %d %d %d\n", my_pid, irr_ngb_i, irr_ngb_j, irr_ngb_k);
    }

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
                    bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
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
                    if (task_check) {
                        calc_t contrib[dof]; vec_zero<idx_t, calc_t, dof>(contrib);// 非规则点可能的贡献
                        idx_t ir = irrg_to_Struct[ptr].loc_id;
                        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                        matvec_mla<idx_t, data_t, calc_t, dof>(irrg_to_Struct[ptr].val, x.irrgPts[ir].val, contrib);

                        kernel_irr(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik, irr_ngb_k - kbeg, contrib);
                        
                        need_to_check = (++ptr) < num_irrgPts_effect;
                        if (need_to_check) {
                            assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                            assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                            irr_ngb_j = irrg_to_Struct[ptr].j;
                        }
                    } else {
                        kernel(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik);
                    }

                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev+1], i_lev, __ATOMIC_RELEASE);
                    else flag[j_lev+1] = i_lev;
                }
                #pragma omp barrier // sync for ptr,need_to_check,irr_ngb_j when each lev done
            }
        }
    }
    else {
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
            const idx_t vec_off = j * vec_dki_size  + i * vec_dk_size  + kbeg * dof;
            const idx_t mat_off = j * mat_edki_size + i * mat_edk_size + kbeg * mat_ed_size;
            const idx_t invD_off= j * invD_dki_size + i * invD_dk_size + kbeg * elms;
            const data_t * L_jik = L_data + mat_off, * U_jik = U_data + mat_off, * invD_jik = invD_data + invD_off;
            const calc_t * sqD_jik = sqD_data ? (sqD_data + vec_off) : nullptr;
            calc_t * x_jik = x_data + vec_off;
            const calc_t * b_jik = b_data + vec_off;
            if (task_check) {
                calc_t contrib[dof]; vec_zero<idx_t, calc_t, dof>(contrib);// 非规则点可能的贡献
                idx_t ir = irrg_to_Struct[ptr].loc_id;
                assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                matvec_mla<idx_t, data_t, calc_t, dof>(irrg_to_Struct[ptr].val, x.irrgPts[ir].val, contrib);

                kernel_irr(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik, irr_ngb_k - kbeg, contrib);

                need_to_check = (++ptr) < num_irrgPts_effect;
                if (need_to_check) {
                    assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                    assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                    irr_ngb_j = irrg_to_Struct[ptr].j;
                }
            } else {
                kernel(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik);
            }
        }
    }
}

template<typename idx_t, typename data_t, typename setup_t, typename calc_t, int dof>
void PointGS<idx_t, data_t, setup_t, calc_t, dof>::BackwardPass(const   par_structVector<idx_t, calc_t, dof> & b, 
                                                                        par_structVector<idx_t, calc_t, dof> & x) const
{
    const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
    const par_structMatrix<idx_t, setup_t, setup_t, dof> * par_A = (par_structMatrix<idx_t, setup_t, setup_t, dof>*)(this->oper);
    assert(LUinvD_separated);
    assert(LUinvD_separated);
    CHECK_LOCAL_HALO(x_vec, b_vec);

    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;
    idx_t num_diag;
    const data_t * L_data = nullptr, * U_data = nullptr;
    idx_t mat_edki_size, mat_edk_size, mat_ed_size;
    if (U_cprs) {
        CHECK_LOCAL_HALO(x_vec, *U_cprs);
        assert(L_cprs);
        num_diag = U_cprs->num_diag / dof;
        L_data = L_cprs->data; U_data = U_cprs->data;
        mat_edki_size = U_cprs->slice_edki_size;
        mat_edk_size  = U_cprs->slice_edk_size ;
        mat_ed_size   = U_cprs->slice_ed_size  ;
    } else {
        CHECK_LOCAL_HALO(x_vec, *U);
        num_diag = (U->num_diag << 1) + 1;
        L_data = L->data; U_data = U->data;
        mat_edki_size = U->slice_edki_size;
        mat_edk_size  = U->slice_edk_size ;
        mat_ed_size   = U->slice_ed_size  ;
    }
    const data_t * invD_data = invD->data;
    const calc_t * sqD_data = sqrt_D ? sqrt_D->data : nullptr;
    const idx_t vec_dk_size = x_vec.slice_dk_size, vec_dki_size = x_vec.slice_dki_size;
    const idx_t invD_dk_size = invD->slice_dk_size, invD_dki_size = invD->slice_dki_size;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t elms = dof*dof;

    const calc_t weight = this->weight;
    const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const calc_t,
        const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*kernel_irr) (const idx_t, const idx_t, const idx_t, const calc_t,
        const data_t*, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*, const idx_t, const calc_t*) = nullptr;
    kernel = this->zero_guess ? AOS_backward_zero : AOS_backward_ALL;
    kernel_irr = this->zero_guess ? AOS_backward_zero_irr : AOS_backward_ALL_irr;
    assert(kernel);
    assert(kernel_irr);

    // 后扫先处理结构点
    idx_t ptr = num_irrgPts_effect - 1, irr_ngb_i = -1, irr_ngb_j = -1, irr_ngb_k = -1;
    bool need_to_check = ptr >= 0;
    if (need_to_check) {
        irr_ngb_i = irrg_to_Struct[ptr].i;  irr_ngb_j = irrg_to_Struct[ptr].j;  irr_ngb_k = irrg_to_Struct[ptr].k;
    }

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
                    bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
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
                    if (task_check) {
                        calc_t contrib[dof]; vec_zero<idx_t, calc_t, dof>(contrib);// 非规则点可能的贡献
                        idx_t ir = irrg_to_Struct[ptr].loc_id;
                        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                        matvec_mla<idx_t, data_t, calc_t, dof>(irrg_to_Struct[ptr].val, x.irrgPts[ir].val, contrib);

                        kernel_irr(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik, irr_ngb_k - kbeg, contrib);

                        need_to_check = (--ptr) >= 0;
                        if (need_to_check) {
                            assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                            assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                            irr_ngb_j = irrg_to_Struct[ptr].j;
                        }
                    } else {
                        kernel(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik);
                    }
                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev], i_lev, __ATOMIC_RELEASE);
                    else flag[j_lev] = i_lev;
                }
                #pragma omp barrier // sync for ptr,need_to_check,irr_ngb_j when each lev done
            }
        }
    }
    else {
        for (idx_t j = jend - 1; j >= jbeg; j--)
        for (idx_t i = iend - 1; i >= ibeg; i--) {
            bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
            const idx_t vec_off = j * vec_dki_size  + i * vec_dk_size  + kend * dof;
            const idx_t mat_off = j * mat_edki_size + i * mat_edk_size + kend * mat_ed_size;
            const idx_t invD_off= j * invD_dki_size + i * invD_dk_size + kend * elms;
            const data_t * L_jik = L_data + mat_off, * U_jik = U_data + mat_off, * invD_jik = invD_data + invD_off;
            const calc_t * sqD_jik = sqD_data ? (sqD_data + vec_off) : nullptr;
            calc_t * x_jik = x_data + vec_off;
            const calc_t * b_jik = b_data + vec_off;
            if (task_check) {
                calc_t contrib[dof]; vec_zero<idx_t, calc_t, dof>(contrib);// 非规则点可能的贡献
                idx_t ir = irrg_to_Struct[ptr].loc_id;
                assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                matvec_mla<idx_t, data_t, calc_t, dof>(irrg_to_Struct[ptr].val, x.irrgPts[ir].val, contrib);

                kernel_irr(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik, irr_ngb_k - kbeg, contrib);
                
                need_to_check = (--ptr) >= 0;
                if (need_to_check) {
                    assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                    assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                    irr_ngb_j = irrg_to_Struct[ptr].j;
                }
            } else {
                kernel(col_height, vec_dk_size, vec_dki_size, weight, L_jik, U_jik, invD_jik, b_jik, x_jik, sqD_jik);
            }
        }
    }

    // 再处理非规则点
    assert(par_A->num_irrgPts == x.num_irrgPts && x.num_irrgPts == b.num_irrgPts);
    for (idx_t ir = par_A->num_irrgPts - 1; ir >= 0; ir--) {
        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid && x.irrgPts[ir].gid == b.irrgPts[ir].gid);
        assert(par_A->irrgPts[ir].gid == irrgPts_invD[ir].gid);
        const idx_t pbeg = par_A->irrgPts[ir].beg, pend = pbeg + par_A->irrgPts[ir].nnz;
        calc_t tmp[dof], tmp2[dof];
        #pragma GCC unroll 4
        for (idx_t f = 0; f < dof; f++)// tmp = -b
            tmp[f] = - b.irrgPts[ir].val[f];
        assert(par_A->irrgPts_ngb_ijk[(pend-1)*3] == -1);// 对角元位置
        data_t * invD_ptr = irrgPts_invD[ir].val;
        #pragma omp parallel for schedule(static) reduction(+:tmp)
        for (idx_t p = pbeg; p < pend - 1; p++) {// 跳过了对角元
            const idx_t ngb_i = par_A->irrgPts_ngb_ijk[p*3  ],
                        ngb_j = par_A->irrgPts_ngb_ijk[p*3+1],
                        ngb_k = par_A->irrgPts_ngb_ijk[p*3+2];// global coord
            const idx_t i = ibeg + ngb_i - par_A->offset_x,
                        j = jbeg + ngb_j - par_A->offset_y,
                        k = kbeg + ngb_k - par_A->offset_z;
            const setup_t * src_ptr = par_A->irrgPts_A_vals + (p<<1)*elms;
            data_t dst_ptr[dof*dof];
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof*dof; f++)
                dst_ptr[f] = src_ptr[f];
            matvec_mla<idx_t, data_t, calc_t, dof>(dst_ptr, x_data + j*vec_dki_size + i*vec_dk_size +  k*dof, tmp);
        }// 此时 tmp = - b + L*x^{t} + U*x^{t}
        matvec_mul<idx_t, data_t, calc_t, dof>(invD_ptr, tmp, tmp2, - weight);// tmp2 = w*D^{-1}*(b - L*x^{t} - U*x^{t})
        #pragma GCC unroll 4
        for (idx_t f = 0; f < dof; f++)
            x.irrgPts[ir].val[f] = (1.0 - weight) * x.irrgPts[ir].val[f] + tmp2[f];
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

    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if constexpr (sizeof(setup_t) != sizeof(data_t)) {
        if (my_pid == 0) 
            printf(" WARNING: PGS Setop truncate setup_t of %ld to data_t of %ld bytes\n", sizeof(setup_t), sizeof(data_t));
    }

    const idx_t jbeg = hy, jend = jbeg + ly,// 注意这里经常写bug
                ibeg = hx, iend = ibeg + lx,
                kbeg = hz, kend = kbeg + lz;
    if (par_A.LU_compressed) {
        const seq_structVector<idx_t, setup_t, dof*dof> * Diag = par_A.Diag;
        L_cprs = new seq_structMatrix<idx_t, data_t, calc_t, 1>(diag_id * dof, lx, ly, lz, hx, hy, hz);
        U_cprs = new seq_structMatrix<idx_t, data_t, calc_t, 1>(*L_cprs);
        #pragma omp parallel 
        {
            setup_t buf[dof * dof], inv_res[dof * dof];
            #pragma omp for collapse(3) schedule(static)
            for (idx_t j = jbeg; j < jend; j++)
            for (idx_t i = ibeg; i < iend; i++)
            for (idx_t k = kbeg; k < kend; k++) {
                // compressed L and U
                idx_t cprs_off = j * L_cprs->slice_edki_size + i * L_cprs->slice_edk_size + k * L_cprs->slice_ed_size;
                for (idx_t d = 0; d < diag_id * dof; d++)
                    L_cprs->data[cprs_off + d] = par_A.L_cprs->data[cprs_off + d];
                for (idx_t d = 0; d < diag_id * dof; d++)
                    U_cprs->data[cprs_off + d] = par_A.U_cprs->data[cprs_off + d];

                // D => calc invD
                const setup_t * Diag_ptr= Diag->data + j * Diag->slice_dki_size + i * Diag->slice_dk_size + k * dof*dof;
                    data_t   * invD_ptr = invD->data + j * invD->slice_dki_size + i * invD->slice_dk_size + k * dof*dof;
                memcpy(buf, Diag_ptr, sizeof(setup_t) * dof * dof);
                matinv_row<idx_t, setup_t, dof>(buf, inv_res);
                for (idx_t p = 0; p < dof*dof; p++)
                    invD_ptr[p] = inv_res[p];
            }
        }
    }
    else {
        L = new seq_structMatrix<idx_t, data_t, calc_t, dof>(diag_id, lx, ly, lz, hx, hy, hz);
        U = new seq_structMatrix<idx_t, data_t, calc_t, dof>(diag_id, lx, ly, lz, hx, hy, hz);
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
    }

    if (par_A.scaled) {
        const seq_structVector<idx_t, setup_t, dof> & src_h = *(par_A.sqrt_D);
        sqrt_D = new seq_structVector<idx_t, calc_t, dof>(lx, ly, lz, hx, hy, hz);
        const idx_t tot_elems = (lx + hx*2) * (ly + hy*2) * (lz + hz*2) * dof;
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < tot_elems; i++)
            sqrt_D->data[i] = src_h.data[i];
    }

    // 处理非规则点
    if (par_A.num_irrgPts > 0) {
        num_irrgPts_invD = par_A.num_irrgPts;
        irrgPts_invD = new IrrgPts_InvMat<idx_t, data_t, dof>[num_irrgPts_invD];
        for (idx_t ir = 0; ir < num_irrgPts_invD; ir++) {
            irrgPts_invD[ir].gid = par_A.irrgPts[ir].gid;
            const idx_t pbeg = par_A.irrgPts[ir].beg, pend = pbeg + par_A.irrgPts[ir].nnz;
            setup_t buf[dof * dof], inv_res[dof * dof];
            memcpy(buf, par_A.irrgPts_A_vals + ((pend-1)<<1)*dof*dof, sizeof(setup_t) * dof * dof);
            matinv_row<idx_t, setup_t, dof>(buf, inv_res);
            for (idx_t p = 0; p < dof*dof; p++)
                irrgPts_invD[ir].val[p] = inv_res[p];
        }
    }

    LUinvD_separated = true;
}

#endif