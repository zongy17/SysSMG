#ifndef SOLID_GMG_COARSEN_HPP
#define SOLID_GMG_COARSEN_HPP

#include "GMG_types.hpp"
#include <string.h>
#include "RAP_3d27.hpp"
#include "RAP_3d7.hpp"

template<typename idx_t, typename data_t, int dof=NUM_DOF>
void XYZ_standard_coarsen(const par_structVector<idx_t, data_t, dof> & fine_vec, const bool periodic[3], 
    COAR_TO_FINE_INFO<idx_t> & coar_to_fine, idx_t stride = 2, idx_t base_x = 0, idx_t base_y = 0, idx_t base_z = 0)
{
    assert(stride == 2);
    // 要求进程内数据大小可以被步长整除
    assert(fine_vec.local_vector->local_x % stride == 0);
    coar_to_fine.fine_base_idx[0] = base_x;
    coar_to_fine.stride[0] = stride;

    assert(fine_vec.local_vector->local_y % stride == 0);
    coar_to_fine.fine_base_idx[1] = base_y;
    coar_to_fine.stride[1] = stride;

    assert(fine_vec.local_vector->local_z % stride == 0);
    coar_to_fine.fine_base_idx[2] = base_z;
    coar_to_fine.stride[2] = stride;

    int my_pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (my_pid == 0)
        printf("coarsen base idx: x %d y %d z %d\n", 
            coar_to_fine.fine_base_idx[0], coar_to_fine.fine_base_idx[1], coar_to_fine.fine_base_idx[2]);
}

template<typename idx_t, typename data_t>
void XY_semi_coarsen(const par_structVector<idx_t, data_t> & fine_vec, const bool periodic[3], 
    COAR_TO_FINE_INFO<idx_t> & coar_to_fine, idx_t stride = 2, idx_t base_x = 0, idx_t base_y = 0)
{
    assert(stride == 2);
    // 要求进程内数据大小可以被步长整除
    assert(fine_vec.local_vector->local_x % stride == 0);
    coar_to_fine.fine_base_idx[0] = base_x;
    coar_to_fine.stride[0] = stride;

    assert(fine_vec.local_vector->local_y % stride == 0);
    coar_to_fine.fine_base_idx[1] = base_y;
    coar_to_fine.stride[1] = stride;

    coar_to_fine.fine_base_idx[2] = 0;// Z方向不做粗化
    coar_to_fine.stride[2] = 1;

    int my_pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (my_pid == 0)
        printf("coarsen base idx: x %d y %d z %d\n", 
            coar_to_fine.fine_base_idx[0], coar_to_fine.fine_base_idx[1], coar_to_fine.fine_base_idx[2]);
}

template<typename idx_t, typename data_t, int dof=NUM_DOF>
par_structMatrix<idx_t, data_t, data_t, dof> * Galerkin_RAP_3d(RESTRICT_TYPE rstr_type, const par_structMatrix<idx_t, data_t, data_t, dof> & fine_mat,
    PROLONG_TYPE prlg_type, const COAR_TO_FINE_INFO<idx_t> & info)
{
    assert( info.stride[0] == info.stride[1] && info.stride[0] == 2);
    assert( info.fine_base_idx[0] == info.fine_base_idx[1] && 
            info.fine_base_idx[1] == info.fine_base_idx[2] && info.fine_base_idx[0] == 0);

    int my_pid; MPI_Comm_rank(fine_mat.comm_pkg->cart_comm, &my_pid);
    if (my_pid == 0) printf("using struct Galerkin(RAP)\n");
    int procs_dims = sizeof(fine_mat.comm_pkg->cart_ids) / sizeof(int); assert(procs_dims == 3);
    int num_procs[3], periods[3], coords[3];
    MPI_Cart_get(fine_mat.comm_pkg->cart_comm, procs_dims, num_procs, periods, coords);

    assert(fine_mat.input_dim[0] == fine_mat.output_dim[0] && fine_mat.input_dim[1] == fine_mat.output_dim[1]
        && fine_mat.input_dim[2] == fine_mat.output_dim[2]);
    const idx_t coar_gx = fine_mat.input_dim[0] / info.stride[0],
                coar_gy = fine_mat.input_dim[1] / info.stride[1],
                coar_gz = fine_mat.input_dim[2] / info.stride[2];
    assert(coar_gx % num_procs[1] == 0 && coar_gy % num_procs[0] == 0 && coar_gz % num_procs[2] == 0);

    par_structMatrix<idx_t, data_t, data_t, dof> * coar_mat = nullptr;
    par_structMatrix<idx_t, data_t, data_t, dof> * padding_mat = nullptr;
    seq_structMatrix<idx_t, data_t, data_t, dof> * fine_mat_local = nullptr;
    seq_structMatrix<idx_t, data_t, data_t, dof> * coar_mat_local = nullptr;
    
    if (fine_mat.num_diag == 7 && ((prlg_type == Plg_linear_8cell && rstr_type == Rst_8cell)
                                || (prlg_type == Plg_linear_4cell && rstr_type == Rst_4cell) )) {// 下一层粗网格也是7对角
    // if (true) {
        coar_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm, 7,
            coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
        fine_mat_local = fine_mat.local_matrix;
        coar_mat_local = coar_mat->local_matrix;
    }
    else {
        if (fine_mat.num_diag != 27) {// 需要补全
            if (my_pid == 0) printf("padding %d to 27\n", fine_mat.num_diag);
            padding_mat = new par_structMatrix<idx_t, data_t, data_t, dof>(fine_mat.comm_pkg->cart_comm, 27,
                fine_mat.input_dim[0], fine_mat.input_dim[1], fine_mat.input_dim[2],
                num_procs[1], num_procs[0], num_procs[2]);
            const seq_structMatrix<idx_t, data_t, data_t, dof> & A_part = *(fine_mat.local_matrix);
                  seq_structMatrix<idx_t, data_t, data_t, dof> & A_full = *(padding_mat->local_matrix);
            CHECK_LOCAL_HALO(A_part, A_full);
            assert(A_full.num_diag == 27);
            const idx_t jbeg = 0, jend = A_full.halo_y * 2 + A_full.local_y;
            const idx_t ibeg = 0, iend = A_full.halo_x * 2 + A_full.local_x;
            const idx_t kbeg = 0, kend = A_full.halo_z * 2 + A_full.local_z;
            const idx_t lms = dof*dof;
            if (A_part.num_diag == 7) {
                #pragma omp parallel for collapse(3) schedule(static)
                for (idx_t j = jbeg; j < jend; j++)
                for (idx_t i = ibeg; i < iend; i++)
                for (idx_t k = kbeg; k < kend; k++) {
#define MATIDX(mat, k, i, j)  (k) * (mat).slice_ed_size + (i) * (mat).slice_edk_size + (j) * (mat).slice_edki_size
                    const data_t * src = A_part.data + MATIDX(A_part, k, i, j);
                    data_t       * dst = A_full.data + MATIDX(A_full, k, i, j);
#undef MATIDX
                    memset(dst          ,           0.0, sizeof(data_t) * lms);
                    memset(dst + 1  *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 2  *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 3  *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 4  *lms, src          , sizeof(data_t) * lms);
                    memset(dst + 5  *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 6  *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 7  *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 8  *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 9  *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 10 *lms, src + 1  *lms, sizeof(data_t) * lms);
                    memset(dst + 11 *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 12 *lms, src + 2  *lms, sizeof(data_t) * lms);
                    memcpy(dst + 13 *lms, src + 3  *lms, sizeof(data_t) * lms);
                    memcpy(dst + 14 *lms, src + 4  *lms, sizeof(data_t) * lms);
                    memset(dst + 15 *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 16 *lms, src + 5  *lms, sizeof(data_t) * lms);
                    memset(dst + 17 *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 18 *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 19 *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 20 *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 21 *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 22 *lms, src + 6  *lms, sizeof(data_t) * lms);
                    memset(dst + 23 *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 24 *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 25 *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 26 *lms,           0.0, sizeof(data_t) * lms);
                }
            }
            else if (A_part.num_diag == 15) {
                #pragma omp parallel for collapse(3) schedule(static)
                for (idx_t j = jbeg; j < jend; j++)
                for (idx_t i = ibeg; i < iend; i++)
                for (idx_t k = kbeg; k < kend; k++) {
#define MATIDX(mat, k, i, j)  (k) * (mat).slice_ed_size + (i) * (mat).slice_edk_size + (j) * (mat).slice_edki_size
                    const data_t * src = A_part.data + MATIDX(A_part, k, i, j);
                    data_t       * dst = A_full.data + MATIDX(A_full, k, i, j);
#undef MATIDX
                    memcpy(dst          , src          , sizeof(data_t) * lms);
                    memcpy(dst + 1  *lms, src + 1  *lms, sizeof(data_t) * lms);
                    memset(dst + 2  *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 3  *lms, src + 2  *lms, sizeof(data_t) * lms);
                    memcpy(dst + 4  *lms, src + 3  *lms, sizeof(data_t) * lms);
                    memset(dst + 5  *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 6  *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 7  *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 8  *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 9  *lms, src + 4  *lms, sizeof(data_t) * lms);
                    memcpy(dst + 10 *lms, src + 5  *lms, sizeof(data_t) * lms);
                    memset(dst + 11 *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 12 *lms, src + 6  *lms, sizeof(data_t) * lms);
                    memcpy(dst + 13 *lms, src + 7  *lms, sizeof(data_t) * lms);
                    memcpy(dst + 14 *lms, src + 8  *lms, sizeof(data_t) * lms);
                    memset(dst + 15 *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 16 *lms, src + 9  *lms, sizeof(data_t) * lms);
                    memcpy(dst + 17 *lms, src + 10 *lms, sizeof(data_t) * lms);
                    memset(dst + 18 *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 19 *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 20 *lms,           0.0, sizeof(data_t) * lms);
                    memset(dst + 21 *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 22 *lms, src + 11 *lms, sizeof(data_t) * lms);
                    memcpy(dst + 23 *lms, src + 12 *lms, sizeof(data_t) * lms);
                    memset(dst + 24 *lms,           0.0, sizeof(data_t) * lms);
                    memcpy(dst + 25 *lms, src + 13 *lms, sizeof(data_t) * lms);
                    memcpy(dst + 26 *lms, src + 14 *lms, sizeof(data_t) * lms);
                }
            }
            else {
                if (my_pid == 0) printf("Error: What should I pad?\n");
                MPI_Abort(MPI_COMM_WORLD, -20230114);
            }
            assert(padding_mat->check_Dirichlet());
            fine_mat_local = padding_mat->local_matrix;
        } else if (fine_mat.num_diag == 27)// 可以直接做
            fine_mat_local = fine_mat.local_matrix;
        
        coar_mat = new par_structMatrix<idx_t, data_t, data_t, dof>(fine_mat.comm_pkg->cart_comm, 27,
                coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
        coar_mat_local = coar_mat->local_matrix;
    }

    // if (rstr_type != Rst_8cell) {
    //     if (my_pid == 0) printf("  using R8 to build Galerkin ...\n");
    //     rstr_type = Rst_8cell;
    // }
    // // assert(prlg_type == Plg_linear_64cell);
    // if (prlg_type != Plg_linear_64cell) {
    //     if (my_pid == 0) printf("  using P64 to build Galerkin ...\n");
    //     prlg_type = Plg_linear_64cell;
    // }
    // assert(fine_mat.num_diag == 27 && coar_mat.num_diag == 27);

    CHECK_HALO(*fine_mat_local, *coar_mat_local);
    const idx_t hx = fine_mat_local->halo_x, hy = fine_mat_local->halo_y, hz = fine_mat_local->halo_z;
    const idx_t f_lx = fine_mat_local->local_x, f_ly = fine_mat_local->local_y, f_lz = fine_mat_local->local_z,
                c_lx = coar_mat_local->local_x, c_ly = coar_mat_local->local_y, c_lz = coar_mat_local->local_z;
    
    // 确定各维上是否是边界
    const bool x_lbdr = fine_mat.comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = fine_mat.comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = fine_mat.comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = fine_mat.comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = fine_mat.comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = fine_mat.comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;

    if (fine_mat.num_diag == 7 && ((prlg_type == Plg_linear_8cell && rstr_type == Rst_8cell)
                                || (prlg_type == Plg_linear_4cell && rstr_type == Rst_4cell) )) {
    // if (true) {
        if (rstr_type == Rst_4cell) {
            if (my_pid == 0) printf("  using \033[1;35m3d7-Galerkin semiXY\033[0m...\n");
            RAP_3d7_semiXY<idx_t, data_t, dof>(fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                    coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                    hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                    x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                    hx, hy, hz
            );
        }
        else {
            if (my_pid == 0) printf("  using \033[1;35m3d7-Galerkin full\033[0m...\n");
            RAP_3d7<idx_t, data_t, dof>(fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                    coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                    hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                    x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                    hx, hy, hz
            );
        }
    } else {
        if (rstr_type == Rst_4cell) {
            assert(prlg_type == Plg_linear_4cell);
            if (my_pid == 0) printf("  using \033[1;35m3d27-Galerkin semiXY\033[0m...\n");
            RAP_3d27_semiXY<idx_t, data_t, dof>(fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                    coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                    hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                    x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                    hx, hy, hz
            );
        }
        else {
            assert(prlg_type == Plg_linear_8cell);
            if (my_pid == 0) printf("  using \033[1;35m3d27-Galerkin full\033[0m...\n");
            RAP_3d27<idx_t, data_t, dof>(fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                    coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                    hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                    x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                    hx, hy, hz
            );
        }
    }

    coar_mat->update_halo();
    // Check if Dirichlet boundary condition met
    assert(coar_mat->check_Dirichlet());
    if (fine_mat.Diags_separated)
        coar_mat->separate_Diags();

    // {// 打印出来看看
    //     int num_proc; MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    //     if (my_pid == 0 && num_proc == 1) {
    //         FILE * fp = fopen("coar_mat.txt", "w+");
    //         const idx_t CX = c_lx + hx * 2,
    //                     CZ = c_lz + hz * 2;
    //         for (idx_t J = hy; J < hy + c_ly; J++)
    //         for (idx_t I = hx; I < hx + c_lx; I++)
    //         for (idx_t K = hz; K < hz + c_lz; K++) {
    //             const data_t * ptr = coar_mat_local.data + coar_mat.num_diag * (K + CZ * (I + CX * J));
    //             for (idx_t d = 0; d < coar_mat.num_diag; d++) 
    //                 fprintf(fp, " %16.10e", ptr[d]);
    //             fprintf(fp, "\n");
    //         }
    //         fclose(fp);
    //     }
    // }

    if (padding_mat != nullptr) {
        delete padding_mat; padding_mat = nullptr;
    }
    return coar_mat;
}

#endif