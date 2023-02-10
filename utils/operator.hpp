#ifndef SOLID_OPERATOR_HPP
#define SOLID_OPERATOR_HPP

// 虚基类Operator，需要子类重写Mult()，如有必要还需重写析构函数
// 算子需要两种精度：数据的存储精度data_t，和计算时的精度calc_t
template<typename idx_t, typename data_t, typename calc_t>
class Operator {
public:
    idx_t  input_dim[3];// 输入向量的全局大小
    idx_t output_dim[3];// 输出向量的全局大小

    explicit Operator(idx_t s = 0) : input_dim{s, s, s}, output_dim{s, s, s} {  }
    Operator(idx_t in0, idx_t in1, idx_t in2, idx_t out0, idx_t out1, idx_t out2) 
        : input_dim{in0, in1, in2}, output_dim{out0, out1, out2} {  }
    virtual ~Operator() {}

    // 截断矩阵/算子元素到16位精度存储（模拟）
    virtual void truncate() = 0;

    // 带零初值优化的接口，由上层调用者决定是否要使用零初值的
    virtual void Mult(const par_structVector<idx_t, calc_t> & input, 
                            par_structVector<idx_t, calc_t> & output, bool use_zero_guess) const = 0;
};

// 虚基类Solver，继承自虚基类Operator
// 需要子类重写SetOperator()和Mult()，如有必要还需重写析构函数
// data_t是数据存储的精度，setup_t是建立算子时的精度，calc_t是算子作用时的精度（对应向量的精度）
// 一般 data_t <= calc_t，以及data_t <= setup_t
template<typename idx_t, typename data_t, typename setup_t, typename calc_t>
class Solver : public Operator<idx_t, data_t, calc_t> {
public:
    mutable bool zero_guess = false;// 初始解是否为0
    mutable data_t weight = 1.0;// 油藏模拟的算例用1.0

    explicit Solver(idx_t s = 0, bool use_zero_guess = false) : Operator<idx_t, data_t, calc_t>(s) {zero_guess = use_zero_guess;} 

    Solver(idx_t in0, idx_t in1, idx_t in2, idx_t out0, idx_t out1, idx_t out2, bool use_zero_guess = false) :
        Operator<idx_t, data_t, calc_t>(in0, in1, in2, out0, out1, out2) {zero_guess = use_zero_guess; }
    virtual void SetOperator(const Operator<idx_t, setup_t, setup_t> & op) = 0;
    virtual void SetRelaxWeight(data_t wt) {weight = wt;}
};

#endif