#ifndef SOLID_ITER_SOLVER_HPP
#define SOLID_ITER_SOLVER_HPP

#include "../utils/operator.hpp"
#include <memory>

typedef enum {PREC, OPER, AXPY, DOT, NUM_KRYLOV_RECORD} KRYLOV_RECORD_TYPES; 

// 虚基类IterativeSolver，支持两种精度
// 继承自虚基类->Solver->Operator，重写了SetOperator()，还需下一层子类重写Mult()
// 预条件的存储类型pc_data_t, 预条件的计算类型pc_calc_t, 外迭代的精度ksp_t
template<typename idx_t, typename pc_data_t, typename pc_calc_t, typename ksp_t>
class IterativeSolver : public Solver<idx_t, ksp_t, ksp_t, ksp_t> {
public:
    // oper(外迭代的算子/矩阵)，可以和prec(预条件子)采用不一样的精度
    const Operator<idx_t, ksp_t, ksp_t> *oper = nullptr;
    // 预条件子的存储采用低精度，计算采用高精度
    Solver<idx_t, pc_data_t, ksp_t, pc_calc_t> *prec = nullptr;

    int max_iter = 10, print_level = -1;
    double rel_tol = 0.0, abs_tol = 0.0;// 用高精度的收敛判断

    // stats
    mutable int final_iter = 0, converged = 0;
    mutable double final_norm;
    mutable double part_times[NUM_KRYLOV_RECORD];

    IterativeSolver() : Solver<idx_t, ksp_t, ksp_t, ksp_t>() {   }

    void SetRelTol(double rtol) { rel_tol = rtol; }
    void SetAbsTol(double atol) { abs_tol = atol; }
    void SetMaxIter(int max_it) { max_iter = max_it; }
    void SetPrintLevel(int print_lvl) { print_level = print_level; }

    int GetNumIterations() const { return final_iter; }
    int GetConverged() const { return converged; }
    double GetFinalNorm() const { return final_norm; }

    /// This should be called before SetOperator
    virtual void SetPreconditioner(Solver<idx_t, pc_data_t, ksp_t, pc_calc_t> & pr) {
        prec = & pr;
        prec->zero_guess = true;// 预条件一般可以用0初值
    }

    /// Also calls SetOperator for the preconditioner if there is one
    virtual void SetOperator(const Operator<idx_t, ksp_t, ksp_t> & op) {
        oper = & op;
        this->input_dim[0] = op.input_dim[0];
        this->input_dim[1] = op.input_dim[1];
        this->input_dim[2] = op.input_dim[2];

        this->output_dim[0] = op.output_dim[0];
        this->output_dim[1] = op.output_dim[1];
        this->output_dim[2] = op.output_dim[2];

        if (prec) {
            int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
            if (my_pid == 0) printf("IterSolver: SetOperator\n");
            prec->SetOperator(*oper);
        }
    }

    virtual void truncate() {
        if (this->prec) {
            int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
            if (my_pid == 0) printf("IterSolver: Truncate\n");
            this->prec->truncate();
        }
    }

    // 迭代法里的点积（默认采用双精度）
    double Dot(const par_structVector<idx_t, ksp_t> & x, const par_structVector<idx_t, ksp_t> & y) const {
        return vec_dot<idx_t, ksp_t, double>(x, y);
    }
    // 迭代法里的范数（默认采用双精度）
    double Norm(const par_structVector<idx_t, ksp_t> & x) const {
        return sqrt(Dot(x, x));
    }

protected:
    virtual void Mult(const par_structVector<idx_t, ksp_t> & input, 
                            par_structVector<idx_t, ksp_t> & output) const = 0;
public:
    // 所有具体的迭代方法的唯一的公共的外部接口
    void Mult(const par_structVector<idx_t, ksp_t> & input, 
                    par_structVector<idx_t, ksp_t> & output, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;// 迭代法的初值是否为0
        this->Mult(input, output);
        this->zero_guess = false;// reset for safety concern
    }
};

#endif