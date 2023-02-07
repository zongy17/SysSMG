#include "utils/par_struct_mat.hpp"
#include "Solver_ls.hpp"


int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int num_procs, my_pid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    int cnt = 1;
    std::string case_name = std::string(argv[cnt++]);
    // 三维划分
    int case_idx   = atoi(argv[cnt++]);
    int case_idy   = atoi(argv[cnt++]);
    int case_idz   = atoi(argv[cnt++]);
    int num_proc_x = atoi(argv[cnt++]);
    int num_proc_y = atoi(argv[cnt++]);
    int num_proc_z = atoi(argv[cnt++]);

    std::string its_name = std::string(argv[cnt++]);
    int restart    = atoi(argv[cnt++]);

    std::string prc_name = "";
    if (argc >= 8)
        prc_name = std::string(argv[cnt++]);

    if (my_pid == 0) printf("\033[1;35mNum Proc along X: %d, along Y: %d, along Z: %d\033[0m \n", num_proc_x, num_proc_y, num_proc_z);
    if (my_pid == 0) printf("Max threads: %d\n", omp_get_max_threads());

    {   IDX_TYPE num_diag = -1;
        if (strcmp(case_name.c_str(), "LASER" ) == 0) {
            assert(sizeof(KSP_TYPE) == 8);
            assert(NUM_DOF == 3);
            num_diag = 7;
        }
        else if (strcmp(case_name.c_str(), "SOLID") == 0) {
            assert(sizeof(KSP_TYPE) == 8);
            assert(NUM_DOF == 3);
            num_diag = 15;
        } else if (strstr(case_name.c_str(), "DEMO")) {
            if      (strstr(case_name.c_str(), "07"))
                num_diag = 7;
            else if (strstr(case_name.c_str(), "15"))
                num_diag = 15;
            else if (strstr(case_name.c_str(), "27"))
                num_diag = 27;
            else MPI_Abort(MPI_COMM_WORLD, -71927);
            if (my_pid == 0) printf("DEMO program of %d\n", num_diag);
        }
        assert(num_diag != -1);
        par_structVector<IDX_TYPE, KSP_TYPE> * x = nullptr, * b = nullptr, * y = nullptr;
        par_structMatrix<IDX_TYPE, KSP_TYPE, KSP_TYPE> * A = nullptr;
        std::string pathname;
        IterativeSolver<IDX_TYPE, PC_DATA_TYPE, PC_CALC_TYPE, KSP_TYPE> * solver = nullptr;
        Solver         <IDX_TYPE, PC_DATA_TYPE, KSP_TYPE, PC_CALC_TYPE> * precond = nullptr;
        std::string data_path = "/storage/hpcauser/zongyi/HUAWEI/SysSMG/data";

        x = new par_structVector<IDX_TYPE, KSP_TYPE          >
                (MPI_COMM_WORLD,      case_idx, case_idy, case_idz, num_proc_x, num_proc_y, num_proc_z, num_diag!=7);
        A = new par_structMatrix<IDX_TYPE, KSP_TYPE, KSP_TYPE>
                (MPI_COMM_WORLD,  num_diag, case_idx, case_idy, case_idz, num_proc_x, num_proc_y, num_proc_z);
        y = new par_structVector<IDX_TYPE, KSP_TYPE>(*x);
        b = new par_structVector<IDX_TYPE, KSP_TYPE>(*x);

        pathname = data_path + "/" + case_name + 
            "/" + std::to_string(case_idx) + "x" + std::to_string(case_idy) + "x" + std::to_string(case_idz);
        if (my_pid == 0) printf("%s\n", pathname.c_str());

        x->set_val(0.0, true);// 迪利克雷边界条件
        b->set_val(0.0, true);
        y->set_val(0.0, true);
        A->set_val(0.0, true);

        if (strstr(case_name.c_str(), "DEMO")) {
            if (A->num_diag == 7) {
                A->set_diag_val(0, -0.125);
                A->set_diag_val(1, -0.125);
                A->set_diag_val(2, -0.125);
                A->set_diag_val(3,  4.0);
                A->set_diag_val(4, -0.125);
                A->set_diag_val(5, -0.125);
                A->set_diag_val(6, -0.125);
            }
            else if (A->num_diag == 15) {
                A->set_diag_val(0, -0.125); A->set_diag_val(1, -0.125);
                A->set_diag_val(2, -0.125); A->set_diag_val(3, -0.125);
                
                A->set_diag_val(4, -0.125); A->set_diag_val(5, -0.125); A->set_diag_val(6, -0.125);
                A->set_diag_val(7,  8.0); 
                A->set_diag_val(8, -0.125); A->set_diag_val(9, -0.125); A->set_diag_val(10,-0.125);
                
                A->set_diag_val(11, -0.125); A->set_diag_val(12, -0.125);
                A->set_diag_val(13, -0.125); A->set_diag_val(14, -0.125);
            }
            else if (A->num_diag == 27) {
                A->set_diag_val(0, -0.125); A->set_diag_val(1, -0.125); A->set_diag_val(2, -0.125);
                A->set_diag_val(3, -0.125); A->set_diag_val(4, -0.125); A->set_diag_val(5, -0.125);
                A->set_diag_val(6, -0.125); A->set_diag_val(7, -0.125); A->set_diag_val(8, -0.125);

                A->set_diag_val(9, -0.125); A->set_diag_val(10,-0.125); A->set_diag_val(11,-0.125);
                A->set_diag_val(12,-0.125); A->set_diag_val(13, 16.0) ; A->set_diag_val(14,-0.125);
                A->set_diag_val(15,-0.125); A->set_diag_val(16,-0.125); A->set_diag_val(17,-0.125);

                A->set_diag_val(18,-0.125); A->set_diag_val(19,-0.125); A->set_diag_val(20,-0.125);
                A->set_diag_val(21,-0.125); A->set_diag_val(22,-0.125); A->set_diag_val(23,-0.125);
                A->set_diag_val(24,-0.125); A->set_diag_val(25,-0.125); A->set_diag_val(26,-0.125);
            }
            A->set_boundary();
            A->update_halo();
            b->set_val(1.0, false);
        } else {
            b->read_data(pathname, "array_b");
            A->read_data(pathname);
        }

        double fine_dot;
        // x->read_data(pathname, "array_x_exact.8");
        fine_dot = vec_dot<IDX_TYPE, KSP_TYPE, double>(*x, *x);
        if (my_pid == 0) printf(" Read data,  (x , x ) = %.27e\n", fine_dot);

        fine_dot = vec_dot<IDX_TYPE, KSP_TYPE, double>(*b, *b);
        if (my_pid == 0) printf(" Read data,  (b , b ) = %.27e\n", fine_dot);

        assert(A->check_Dirichlet());

#ifdef WRITE_AOS
        A->write_struct_AOS_bin(pathname, "mat.AOS.bin");
        b->write_data(pathname, "b.AOS.bin");
        x->write_data(pathname, "x.AOS.bin");
        // A->write_CSR_bin();
        // b->write_CSR_bin("b");
        // x->write_CSR_bin("x0");
#endif

        A->Mult(*b, *y, false);

        fine_dot = vec_dot<IDX_TYPE, KSP_TYPE, double>(*y, *y);
        if (my_pid == 0) printf(" (Ab, Ab) = %.27e\n", fine_dot);

        // {// check performance for __fp16
        //     par_structMatrix<IDX_TYPE, __fp16, float> A_low
        //         (MPI_COMM_WORLD,  num_diag, case_idx, case_idy, case_idz, num_proc_x, num_proc_y, num_proc_z);
        //     int tot_len = A->num_diag * A->local_matrix->elem_size
        //             * (A->local_matrix->local_x + A->local_matrix->halo_x * 2)
        //             * (A->local_matrix->local_y + A->local_matrix->halo_y * 2)
        //             * (A->local_matrix->local_z + A->local_matrix->halo_z * 2);
        //     #pragma omp parallel for schedule(static)
        //     for (int i = 0; i < tot_len; i++)
        //         A_low.local_matrix->data[i] = A->local_matrix->data[i];
        //     assert(A_low.check_Dirichlet());
        //     // A_low.separate_Diags();
        //     par_structVector<IDX_TYPE, float> 
        //         b_low(MPI_COMM_WORLD, case_idx, case_idy, case_idz, num_proc_x, num_proc_y, num_proc_z, num_diag!=7),
        //         y_low(b_low);
        //     tot_len /= (A->num_diag * NUM_DOF);
        //     #pragma omp parallel for schedule(static)
        //     for (int i = 0; i < tot_len; i++)
        //         b_low.local_vector->data[i] = b->local_vector->data[i];
        //     A_low.Mult(b_low, y_low, false);
        //     fine_dot = vec_dot<IDX_TYPE, float, double>(y_low, y_low);
        //     if (my_pid == 0) printf(" (Ab, Ab) = %.27e\n", fine_dot);
        // }

        // MPI_Barrier(MPI_COMM_WORLD);
        // MPI_Abort(MPI_COMM_WORLD, -20230205);

        if (strstr(prc_name.c_str(), "PGS")) {
            SCAN_TYPE type = SYMMETRIC;
            if      (strstr(prc_name.c_str(), "F")) type = FORWARD;
            else if (strstr(prc_name.c_str(), "B")) type = BACKWARD;
            else if (strstr(prc_name.c_str(), "S")) type = SYMMETRIC;
            if (my_pid == 0) printf("  using \033[1;35mpointwise-GS %d\033[0m as preconditioner\n", type);
            precond = new PointGS<IDX_TYPE, PC_DATA_TYPE, KSP_TYPE, PC_CALC_TYPE>(type);
        } else if (prc_name == "GMG") {
            IDX_TYPE num_discrete = atoi(argv[cnt++]);
            IDX_TYPE num_Galerkin = atoi(argv[cnt++]);
            std::unordered_map<std::string, RELAX_TYPE> trans_smth;
            trans_smth["PGS"]= PGS;
            trans_smth["LU"] = GaussElim;
            std::vector<RELAX_TYPE> rel_types;
            for (IDX_TYPE i = 0; i < num_discrete + num_Galerkin + 1; i++) {
                rel_types.push_back(trans_smth[argv[cnt++]]);
                // if (my_pid == 0) printf("i %d type %d\n", i, rel_types[i]);
            }
            precond = new GeometricMultiGrid<IDX_TYPE, PC_DATA_TYPE, KSP_TYPE, PC_CALC_TYPE>
                (num_discrete, num_Galerkin, {}, rel_types);
        } else {
            if (my_pid == 0) printf("NO preconditioner was set.\n");
        }

        if (its_name == "CG") {
            solver = new CGSolver<IDX_TYPE, PC_DATA_TYPE, PC_CALC_TYPE, KSP_TYPE>();
        } else if (its_name == "GMRES") {
            solver = new GMRESSolver<IDX_TYPE, PC_DATA_TYPE, PC_CALC_TYPE, KSP_TYPE>();
            ((GMRESSolver<IDX_TYPE, PC_DATA_TYPE, PC_CALC_TYPE, KSP_TYPE>*)solver)->SetRestartlen(restart);
        } else {
            if (my_pid == 0) printf("INVALID iterative solver name of %s\nOnly GCR, CG, GMRES, FGMRES available\n", its_name.c_str());
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        solver->SetMaxIter(30);
        if      (strcmp(case_name.c_str(), "LASER" ) == 0) solver->SetRelTol(1e-9);
        else if (strcmp(case_name.c_str(), "SOLID" ) == 0) solver->SetRelTol(1e-9);
        if (precond != nullptr)
            solver->SetPreconditioner(*precond);
        solver->SetOperator(*A);

#ifdef TRC
        if (my_pid == 0) printf("\033[1;35mTruncate to __fp16...\033[0m\n");
        solver->truncate();
#endif

#ifdef STAT_NNZ
        {
            int num_inter = 25;
            double lb[num_inter], ub[num_inter];
            int nnz_cnt[num_inter];
            lb[0] = 1e-22;
            for (int i = 0; i < num_inter; i++) {
                ub[i] = lb[i] * 10.0;
                if (i < num_inter - 1)
                    lb[i + 1] = ub[i];
                nnz_cnt[i] = 0;
            }
            int zero_cnt = 0;
            int num_theta = 25;
            double lb_theta[num_theta], ub_theta[num_theta];
            int theta_cnt[num_theta];
            ub_theta[num_theta-1] = 1.0;
            for (int i = num_theta-1; i >= 0; i--) {
                lb_theta[i] = ub_theta[i] / 10.0;
                theta_cnt[i] = 0;
                if (i > 0)
                    ub_theta[i-1] = lb_theta[i];
            }

            assert(num_procs == 1);
            const int jbeg = A->local_matrix->halo_y, jend = jbeg + A->local_matrix->local_y;
            const int ibeg = A->local_matrix->halo_x, iend = ibeg + A->local_matrix->local_x;
            const int kbeg = A->local_matrix->halo_z, kend = kbeg + A->local_matrix->local_z;
            const int nd = A->num_diag; assert(nd % 2 == 1);
            const int diag_id = (nd - 1) / 2;
            const int dof = 3; assert(dof*dof == A->local_matrix->elem_size);
            int tot_nnz = 0;
            for (int j = jbeg; j < jend; j++)
            for (int i = ibeg; i < iend; i++)
            for (int k = kbeg; k < kend; k++) 
            for (int f = 0; f < dof; f++) {
                const int nnz_per_row = nd * dof;
                KSP_TYPE abs_nnz[nnz_per_row];
                const KSP_TYPE * Adata = A->local_matrix->data + j * A->local_matrix->slice_edki_size 
                        + i * A->local_matrix->slice_edk_size + k * A->local_matrix->slice_ed_size;
                for (int d = 0; d < nd; d++) {
                    const KSP_TYPE* ptr = Adata + d * A->local_matrix->elem_size + f * dof;
                    for (int g = 0; g < dof; g++)
                        abs_nnz[d*dof + g] = fabs(ptr[g]);
                }
                
                for (int g = 0; g < nnz_per_row; g++) {// 遍历该行所有非零元
                    if (abs_nnz[g] == 0.0) {
                        zero_cnt++;
                        continue;
                    }
                    for (int t = 0; t < num_inter; t++) {
                        if (abs_nnz[g] >= lb[t] && abs_nnz[g] < ub[t]) {
                            nnz_cnt[t] ++;
                            break;
                        }
                    }
                }
                // 计算该行的theta值
                double max_offd = 0.0, min_offd = 1e30;
                for (int g = 0; g < nnz_per_row; g++) {
                    if (abs_nnz[g] == 0.0) continue;
                    tot_nnz ++;
                    if (g == diag_id*dof + f) continue;
                    max_offd = std::max(max_offd, abs_nnz[g]);
                    min_offd = std::min(min_offd, abs_nnz[g]);
                }
                double theta = min_offd / max_offd;
                for (int t = 0; t < num_theta; t++) {
                    if (theta >= lb_theta[t] && theta < ub_theta[t]) {
                        theta_cnt[t] ++;
                        break;
                    }
                }
            }
            printf("Offd nnz\n");
            for (int i = 0; i < num_inter; i++) {
                double ratio = (double) nnz_cnt[i] / (double) tot_nnz;
                printf("[%.2e,%.2e): %d %.6f\n", lb[i], ub[i], nnz_cnt[i], ratio);
            }
            printf("theta\n");
            int tot_ndof = (jend - jbeg) * (iend - ibeg) * (kend - kbeg) * dof;
            for (int i = 0; i < num_theta; i++) {
                double ratio = (double) theta_cnt[i] / (double) tot_ndof;
                printf("[%.2e,%.2e): %d %.6f\n", lb_theta[i], ub_theta[i], theta_cnt[i], ratio);
            }
        }
#endif

        // if (prc_name == "GMG") {
        //     assert(num_Galerkin > 0);
        //     KSP_TYPE  wgts[num_Galerkin+1];
        //     for (int i = 0; i < num_Galerkin+1; i++) wgts[i] = 1.12;
        //     ((GeometricMultiGrid<IDX_TYPE, PC_TYPE, KSP_TYPE, KSP_TYPE>*)precond)->SetRelaxWeights(wgts, num_Galerkin+1);
        // }

        double t1 = wall_time();
        solver->Mult(*b, *x, false);
        t1 = wall_time() - t1;
        if (my_pid == 0) printf("Solve costs %.6f s\n", t1);
        // x->write_data(pathname, "array_x_exact." + std::to_string(solver->final_iter));
        double min_times[NUM_KRYLOV_RECORD], max_times[NUM_KRYLOV_RECORD], avg_times[NUM_KRYLOV_RECORD];
        MPI_Allreduce(solver->part_times, min_times, NUM_KRYLOV_RECORD, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(solver->part_times, max_times, NUM_KRYLOV_RECORD, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(solver->part_times, avg_times, NUM_KRYLOV_RECORD, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
        for (int i = 0; i < NUM_KRYLOV_RECORD; i++)
            avg_times[i] /= num_procs;
        if (my_pid == 0) {
            printf("prec time min/avg/max %.3e %.3e %.3e\n", min_times[PREC], avg_times[PREC], max_times[PREC]);
            printf("oper time min/avg/max %.3e %.3e %.3e\n", min_times[OPER], avg_times[OPER], max_times[OPER]);
            printf("axpy time min/avg/max %.3e %.3e %.3e\n", min_times[AXPY], avg_times[AXPY], max_times[AXPY]);
            printf("dot  tune min/avg/max %.3e %.3e %.3e\n", min_times[DOT ], avg_times[DOT ], max_times[DOT ]);
        }

        A->Mult(*x, *y, false);
        vec_add(*b, -1.0, *y, *y);
        double true_r_norm = vec_dot<IDX_TYPE, KSP_TYPE, double>(*y, *y);
        true_r_norm = sqrt(true_r_norm);

        double b_norm = vec_dot<IDX_TYPE, KSP_TYPE, double>(*b, *b);
        b_norm = sqrt(b_norm);
         if (my_pid == 0) printf("\033[1;35mtrue ||r|| = %20.16e ||r||/||b||= %20.16e\033[0m\n", 
            true_r_norm, true_r_norm / b_norm);

        if (b != nullptr) {delete b; b = nullptr;}
        if (x != nullptr) {delete x; x = nullptr;}
        if (y != nullptr) {delete y; y = nullptr;}
        if (A != nullptr) {delete A; A = nullptr;}
        if (solver != nullptr) {delete solver; solver = nullptr;}
        if (precond != nullptr) {delete precond; precond = nullptr;}
    }

    MPI_Finalize();
    return 0;
}