#ifndef SOLID_CG_HPP
#define SOLID_CG_HPP

#include "iter_solver.hpp"
#include "../utils/par_struct_mat.hpp"
#define DOT_FISSON

template<typename idx_t, typename pc_data_t, typename pc_calc_t, typename ksp_t>
class CGSolver : public IterativeSolver<idx_t, pc_data_t, pc_calc_t, ksp_t> {
public:

    CGSolver() {  };
    
    virtual void Mult(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const ;
};

template<typename idx_t, typename pc_data_t, typename pc_calc_t, typename ksp_t>
void CGSolver<idx_t, pc_data_t, pc_calc_t, ksp_t>::Mult(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const 
{
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);

    MPI_Comm comm = x.comm_pkg->cart_comm;
    int my_pid; MPI_Comm_rank(comm, &my_pid);
    double * record = this->part_times;
    memset(record, 0, sizeof(double) * NUM_KRYLOV_RECORD);

    // 初始化辅助向量：r残差，p搜索方向，Ap为A乘以搜索方向
    par_structVector<idx_t, ksp_t> r(x), Ap(x), p(x);
#if KSP_BIT!=PC_CALC_BIT
    par_structVector<idx_t, pc_calc_t> * pc_buf_b = nullptr, * pc_buf_x = nullptr;
    const idx_t num_diag = ((const par_structMatrix<idx_t, ksp_t, ksp_t>*)this->oper)->num_diag;
    pc_buf_b = new par_structVector<idx_t, pc_calc_t>(r.comm_pkg->cart_comm,
        r.global_size_x, r.global_size_y, r.global_size_z,
        r.global_size_x/r.local_vector->local_x,
        r.global_size_y/r.local_vector->local_y,
        r.global_size_z/r.local_vector->local_z, num_diag != 7);
    pc_buf_x = new par_structVector<idx_t, pc_calc_t>(*pc_buf_b);
    pc_buf_b->set_halo(0.0);
    pc_buf_x->set_halo(0.0);
    
    const idx_t jbeg = r.local_vector->halo_y, jend = jbeg + r.local_vector->local_y,
                ibeg = r.local_vector->halo_x, iend = ibeg + r.local_vector->local_x,
                kbeg = r.local_vector->halo_z, kend = kbeg + r.local_vector->local_z;
    const idx_t col_height = kend - kbeg;
    const idx_t vec_dki_size = r.local_vector->slice_dki_size, vec_dk_size = r.local_vector->slice_dk_size;
#endif
    r.set_halo(0.0);
    Ap.set_halo(0.0);
    p.set_halo(0.0);

    this->oper->Mult(x, r, false);
    vec_add(b, -1.0, r, r);

    double norm_b = this->Norm(b);
    double norm_r = this->Norm(r);
    double alpha_nom, alpha_denom, beta_nom;

    if (this->prec) {// 有预条件子，则以预条件后的残差M^{-1}*r作为搜索方向 p = M^{-1}*r
#if KSP_BIT!=PC_CALC_BIT
        {// copy in
            const seq_structVector<idx_t, ksp_t> & src = *(r.local_vector);
            seq_structVector<idx_t, pc_calc_t> & dst = *(pc_buf_b->local_vector);
            CHECK_LOCAL_HALO(src, dst);
            #pragma omp parallel for collapse(2) schedule(static)
            for (idx_t j = jbeg; j < jend; j++)
            for (idx_t i = ibeg; i < iend; i++) {
                const double* src_ptr = src.data + j * vec_dki_size + i * vec_dk_size + kbeg * NUM_DOF;
                float       * dst_ptr = dst.data + j * vec_dki_size + i * vec_dk_size + kbeg * NUM_DOF;
                const idx_t tot_len = col_height * NUM_DOF;
                for (idx_t k = 0; k < tot_len; k++)
                    dst_ptr[k] = src_ptr[k];
            }
        }
        pc_buf_x->set_val(0.0);
        record[PREC] -= wall_time();
        this->prec->Mult(*pc_buf_b, *pc_buf_x, true);
        record[PREC] += wall_time();
        {// copy out
            const seq_structVector<idx_t, pc_calc_t> & src = *(pc_buf_x->local_vector);
            seq_structVector<idx_t, ksp_t> & dst = *(p.local_vector);
            CHECK_LOCAL_HALO(src, dst);
            #pragma omp parallel for collapse(2) schedule(static)
            for (idx_t j = jbeg; j < jend; j++)
            for (idx_t i = ibeg; i < iend; i++) {
                const float* src_ptr = src.data + j * vec_dki_size + i * vec_dk_size + kbeg * NUM_DOF;
                double     * dst_ptr = dst.data + j * vec_dki_size + i * vec_dk_size + kbeg * NUM_DOF;
                const idx_t tot_len = col_height * NUM_DOF;
                for (idx_t k = 0; k < tot_len; k++)
                    dst_ptr[k] = src_ptr[k];
            }
        }
#else
        p.set_val(0.0);// 预条件解的残差方程M*p=r的初值设为p=0
        record[PREC] -= wall_time();
        this->prec->Mult(r, p, true);
        record[PREC] += wall_time();
#endif
    } else {// 没有预条件则直接以残差r作为搜索方向
        vec_copy(r, p);
    }

    // Ap = A * p
    this->oper->Mult(p, Ap, false);

    alpha_nom   = this->Dot(r,  p);
    alpha_denom = this->Dot(p, Ap);

    int & iter = this->final_iter;
    if (my_pid == 0) printf("iter %4d   ||b|| = %.16e ||r||/||b|| = %.16e\n", iter, norm_b, norm_r / norm_b);

    for ( ; iter < this->max_iter; ) {
        double alpha = alpha_nom / alpha_denom;
        // 更新解和残差
        vec_add(x,  alpha,  p, x);
        vec_add(r, -alpha, Ap, r);

        // 预条件子
        if (this->prec) {
#if KSP_BIT!=PC_CALC_BIT
            {// copy in
                const seq_structVector<idx_t, ksp_t> & src = *(r.local_vector);
                seq_structVector<idx_t, pc_calc_t> & dst = *(pc_buf_b->local_vector);
                CHECK_LOCAL_HALO(src, dst);
                #pragma omp parallel for collapse(2) schedule(static)
                for (idx_t j = jbeg; j < jend; j++)
                for (idx_t i = ibeg; i < iend; i++) {
                    const double* src_ptr = src.data + j * vec_dki_size + i * vec_dk_size + kbeg * NUM_DOF;
                    float       * dst_ptr = dst.data + j * vec_dki_size + i * vec_dk_size + kbeg * NUM_DOF;
                    const idx_t tot_len = col_height * NUM_DOF;
                    for (idx_t k = 0; k < tot_len; k++)
                        dst_ptr[k] = src_ptr[k];
                }
            }
            pc_buf_x->set_val(0.0);
            record[PREC] -= wall_time();
            this->prec->Mult(*pc_buf_b, *pc_buf_x, true);
            record[PREC] += wall_time();
            {// copy out
                const seq_structVector<idx_t, pc_calc_t> & src = *(pc_buf_x->local_vector);
                seq_structVector<idx_t, ksp_t> & dst = *(Ap.local_vector);
                CHECK_LOCAL_HALO(src, dst);
                #pragma omp parallel for collapse(2) schedule(static)
                for (idx_t j = jbeg; j < jend; j++)
                for (idx_t i = ibeg; i < iend; i++) {
                    const float* src_ptr = src.data + j * vec_dki_size + i * vec_dk_size + kbeg * NUM_DOF;
                    double     * dst_ptr = dst.data + j * vec_dki_size + i * vec_dk_size + kbeg * NUM_DOF;
                    const idx_t tot_len = col_height * NUM_DOF;
                    for (idx_t k = 0; k < tot_len; k++)
                        dst_ptr[k] = src_ptr[k];
                }
            }
#else
            Ap.set_val(0.0);
            record[PREC] -= wall_time();
            this->prec->Mult(r, Ap, true);
            record[PREC] += wall_time();
#endif
        } else {
            vec_copy(r, Ap);
        }
        // 此时tmp存储的量为M^{-1}*r
        iter++;// 执行一次预条件子就算一次迭代

        double tmp_loc[2], tmp_glb[2];// 将两次点积合并到一次全局集合通信
#ifdef DOT_FISSON  
        tmp_loc[0] = seq_vec_dot<idx_t, ksp_t, double>(*(r.local_vector), *(r.local_vector));
        tmp_loc[1] = seq_vec_dot<idx_t, ksp_t, double>(*(r.local_vector), *(Ap.local_vector));
#else
        {
            const seq_structVector<idx_t, ksp_t>& r_loc = *(r.local_vector),
                                                &Ap_loc = *(Ap.local_vector);
            CHECK_LOCAL_HALO(Ap_loc, r_loc);
            const idx_t xbeg = r_loc.halo_x, xend = xbeg + r_loc.local_x,
                        ybeg = r_loc.halo_y, yend = ybeg + r_loc.local_y,
                        zbeg = r_loc.halo_z, zend = zbeg + r_loc.local_z;
            const idx_t slice_dk_size = r_loc.slice_dk_size, slice_dki_size = r_loc.slice_dki_size;
            const idx_t dof = slice_dk_size / (2 * r_loc.halo_z + r_loc.local_z);
            double dot_rr = 0.0, dot_rAp = 0.0;
            #pragma omp parallel for collapse(2) reduction(+:dot_rr,dot_rAp) schedule(static)
            for (idx_t j = ybeg; j < yend; j++)
            for (idx_t i = xbeg; i < xend; i++) {
                idx_t ji_loc = j * slice_dki_size + i * slice_dk_size;
                const ksp_t *Ap_data = Ap_loc.data + ji_loc,
                            * r_data =  r_loc.data + ji_loc;
                for (idx_t k = zbeg; k < zend; k++) {
                    #pragma GCC unroll (4)
                    for (idx_t f = 0; f < dof; f++) {
                        idx_t p = k * dof + f;
                        dot_rr += (double) r_data[p] * (double) r_data[p];
                        dot_rAp+= (double) r_data[p] * (double)Ap_data[p];
                    }
                }
            }
            tmp_loc[0] = dot_rr;
            tmp_loc[1] = dot_rAp;
        }
#endif
        MPI_Allreduce(tmp_loc, tmp_glb, 2, MPI_DOUBLE, MPI_SUM, r.comm_pkg->cart_comm);
    
        // 判敛
        norm_r = sqrt(tmp_glb[0]);
        if (my_pid == 0) printf("iter %4d   alpha %.16e   ||r||/||b|| = %.16e\n", iter, alpha, norm_r / norm_b);
        if (norm_r / norm_b <= this->rel_tol || norm_r <= this->abs_tol) {
            this->converged = 1;
            break;
        }

        beta_nom = tmp_glb[1];// beta的分子
        if (beta_nom < 0.0) {
            if (my_pid == 0) printf("WARNING: PCG: The preconditioner is not positive definite. (Br, r) = %.5e\n", (double)beta_nom);
            // this->converged = 0;
            // break;
        }

        // 计算β并更新搜索方向
        double beta = beta_nom / alpha_nom;
        // 更新搜索方向
        vec_add(Ap, beta, p, p);
        
        // Ap = A * p
        this->oper->Mult(p, Ap, false);

        alpha_denom = this->Dot(p, Ap);
        if (alpha_denom <= 0.0) {
            double dd = this->Dot(p, p);
            if (dd > 0.0) if (my_pid == 0) printf("WARNING: PCG: The operator is not positive definite. (Ad, d) = %.5e\n", (double)alpha_denom);
            if (alpha_denom == 0.0) break;
        }
        alpha_nom = beta_nom;
    }
#if KSP_BIT!=PC_CALC_BIT
    if (pc_buf_x) delete pc_buf_x;
    if (pc_buf_b) delete pc_buf_b;
#endif
}

/*
// 看起来本来就是抄的hypre的
template<typename idx_t, typename ksp_t, typename pc_t>
void CGSolver<idx_t, ksp_t, pc_t>::Mult(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const
{
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);

    MPI_Comm comm = x.comm_pkg->cart_comm;
    int my_pid; MPI_Comm_rank(comm, &my_pid);

    // 初始化辅助向量：r残差，p搜索方向，Ap为A乘以搜索方向
    par_structVector<idx_t, ksp_t> r(x), Ap(x), p(x);
    r.set_halo(0.0);
    Ap.set_halo(0.0);
    p.set_halo(0.0);

    double b_norm = this->Norm(b, b);

    // r = b - A*x
    this->oper->Mult(x, r, false);
    vec_add(b, -1.0, r, r);

    if (this->prec) {
        p.set_val(0.0);
        this->prec->Mult(r, p, true);
    } else {
        vec_copy(r, p);
    }

    double gamma = this->Dot(r, p);
    if (gamma != 0.0) {
        double ieee_check = gamma / gamma;// inf -> nan
        assert(gamma == gamma);
        assert(ieee_check == ieee_check);// nan
    }
    double r_norm = this->Norm(r, r);
    double alpha, gamma_old;

    if (my_pid == 0) {
        hypre_printf("Iters       ||r||_2      ||r||_2/||b||_2\n");
        hypre_printf("-----    ------------    ------------ \n");
    }

    int i = 0;
    while ((i+1) <= this->max_iter) {
        i++;

        this->oper->Mult(p, Ap, false);
        double Apdotp = this->Dot(Ap, p);
        assert(Apdotp != 0.0);
        alpha = gamma / Apdotp;
        gamma_old = gamma;

        vec_add(x,  alpha,  p, x);// x = x + alpha*p
        vec_add(r, -alpha, Ap, r);// r = r - alpha*Ap

        // Ap = M^{-1}*r 
        if (this->prec) {
            Ap.set_val(0.0);
            this->prec->Mult(r, Ap, true);
        } else {
            vec_copy(r, Ap);
        }

        gamma = this->Dot(r, Ap);
        r_norm = this->Norm(r, r);
        if (my_pid == 0) {
            hypre_printf("% 5d    %.5e    %.5e\n", i, r_norm, r_norm / b_norm);
        }
    }

}
*/

#endif