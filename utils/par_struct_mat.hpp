#ifndef SOLID_PAR_STRUCT_MV_HPP
#define SOLID_PAR_STRUCT_MV_HPP

#include "common.hpp"
#include "par_struct_vec.hpp"
#include "operator.hpp"
#include <string>

template<typename idx_t>
struct irrgPts_mat
{
    idx_t gid;// 非规则点的全局索引
    idx_t nnz, beg;// 每个非规则点的非零元和在ngb_ijk, _val中的起始位置
};

template<typename idx_t, typename data_t, typename calc_t, int dof=NUM_DOF>
class par_structMatrix : public Operator<idx_t, data_t, calc_t>  {
public:
    idx_t num_diag;
    const idx_t * stencil = nullptr;
    idx_t offset_x     , offset_y     , offset_z     ;// 该矩阵在全局中的偏移
    bool scaled = false, own_sqrt_D = false;
    seq_structVector<idx_t, calc_t, dof> * sqrt_D = nullptr;

    seq_structMatrix<idx_t, data_t, calc_t, dof> * local_matrix;
    mutable bool LU_compressed = false;
    mutable seq_structVector<idx_t, data_t, dof*dof> * Diag = nullptr;
    mutable seq_structMatrix<idx_t, data_t, calc_t, 1> *L_cprs = nullptr, * U_cprs = nullptr;// 压缩后L和U仅保留对角线的的数据

    // 通信相关的
    StructCommPackage * comm_pkg = nullptr;
    bool own_comm_pkg = false;

    idx_t num_irrgPts = 0;// 非规则点的数目
    irrgPts_mat<idx_t> * irrgPts = nullptr;
    idx_t * irrgPts_ngb_ijk = nullptr;
    data_t * irrgPts_A_vals = nullptr;

    par_structMatrix(MPI_Comm comm, idx_t num_d, idx_t gx, idx_t gy, idx_t gz, idx_t num_proc_x, idx_t num_proc_y, idx_t num_proc_z);
    // 按照model的规格生成一个结构化向量，浅拷贝通信包
    par_structMatrix(const par_structMatrix & model);
    ~par_structMatrix();

    void setup_cart_comm(MPI_Comm comm, idx_t px, idx_t py, idx_t pz, bool unblk);
    void setup_comm_pkg(bool need_corner=true);

    void init_irrPts(const std::string pathname);
    void truncate() {
        int my_pid; MPI_Comm_rank(comm_pkg->cart_comm, &my_pid);
        if (my_pid == 0) printf("Warning: parMat truncated to __fp16 and sqrt_D to f32\n");
        local_matrix->truncate();
#ifdef __aarch64__
        if (LU_compressed) {
            // for (idx_t id = 0; id < num_diag; id++) {// 逐条对角线截断
            //     const idx_t nelem = (Diags[id]->local_x + Diags[id]->halo_x * 2) * (Diags[id]->local_y + Diags[id]->halo_y * 2)
            //                     *   (Diags[id]->local_z + Diags[id]->halo_z * 2);
            //     for (idx_t p = 0; p < nelem * dof*dof; p++) {
            //         __fp16 tmp = (__fp16) Diags[id]->data[p];
            //         Diags[id]->data[p] = (data_t) tmp;
            //     }
            // }
        }
        if (sqrt_D != nullptr) {
            const idx_t sqD_len = (sqrt_D->local_x + sqrt_D->halo_x * 2) * (sqrt_D->local_y + sqrt_D->halo_y * 2)
                                * (sqrt_D->local_z + sqrt_D->halo_z * 2) * dof;
            for (idx_t p = 0; p < sqD_len; p++) {
                float tmp = (float) sqrt_D->data[p];
                sqrt_D->data[p] = (data_t) tmp;
            }
        }
        // 非规则点也截断
        idx_t d2 = dof*dof;
        for (idx_t ir = 0; ir < num_irrgPts; ir++) {
            idx_t pbeg = irrgPts[ir].beg, pend = pbeg + irrgPts[ir].nnz;
            for (idx_t p = pbeg; p < pend; p++) {
                for (idx_t f = 0; f < d2; f++) {
                    __fp16 tmp = (__fp16) irrgPts_A_vals[(p<<1)*d2+f];
                    irrgPts_A_vals[(p<<1)*d2+f] = (data_t) tmp;
                }
                for (idx_t f = 0; f < d2; f++) {
                    __fp16 tmp = (__fp16) irrgPts_A_vals[((p<<1)+1)*d2+f];
                    irrgPts_A_vals[((p<<1)+1)*d2+f] = (data_t) tmp;
                }
            }
        }
#else
        printf("architecture not support truncated to fp16\n");
#endif
    }
    void compress_LU() const;
    void compress_Mult(const seq_structVector<idx_t, calc_t, dof> & x, seq_structVector<idx_t, calc_t, dof> & y) const ;
    void (*compress_spmv)(const idx_t, const idx_t, const idx_t, const data_t*, const data_t*, const data_t*,
        const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*compress_spmv_scaled)(const idx_t, const idx_t, const idx_t, const data_t*, const data_t*, const data_t*,
        const calc_t*, calc_t*, const calc_t*) = nullptr;

    void update_halo();
    void Mult(const par_structVector<idx_t, calc_t, dof> & x, 
                    par_structVector<idx_t, calc_t, dof> & y, bool use_zero_guess/* ignored */) const {
        Mult(x, y);
    } 
protected:
    void Mult(const par_structVector<idx_t, calc_t, dof> & x, par_structVector<idx_t, calc_t, dof> & y) const ;

public:
    void read_data(const std::string pathname);
    void set_val(data_t val, bool halo_set=false);
    void set_diag_val(idx_t d, data_t val);
    void init_random_base2(idx_t max_power = 6);

    bool check_Dirichlet();
    void set_boundary();
    void scale(const data_t scaled_diag);
    bool check_scaling(const data_t scaled_diag);
    void copy_irrgPts(const par_structMatrix & src);
    void write_struct_AOS_bin(const std::string pathname, const std::string file);
    void write_CSR_bin() const ;
};


/*
 * * * * * par_structMatrix * * * * *  
 */

template<typename idx_t, typename data_t, typename calc_t, int dof>
par_structMatrix<idx_t, data_t, calc_t, dof>::par_structMatrix(MPI_Comm comm, idx_t num_d,
    idx_t global_size_x, idx_t global_size_y, idx_t global_size_z, idx_t num_proc_x, idx_t num_proc_y, idx_t num_proc_z)
    : Operator<idx_t, data_t, calc_t>(global_size_x, global_size_y, global_size_z, global_size_x, global_size_y, global_size_z), 
        num_diag(num_d)
{
    // for GMG concern: must be fully divided by processors
    assert(global_size_x % num_proc_x == 0);
    assert(global_size_y % num_proc_y == 0);
    assert(global_size_z % num_proc_z == 0);

    setup_cart_comm(comm, num_proc_x, num_proc_y, num_proc_z, false);

    int (&cart_ids)[3] = comm_pkg->cart_ids;
    offset_y = cart_ids[0] * global_size_y / num_proc_y;
    offset_x = cart_ids[1] * global_size_x / num_proc_x;
    offset_z = cart_ids[2] * global_size_z / num_proc_z;

    // 建立本地数据的内存
    local_matrix = new seq_structMatrix<idx_t, data_t, calc_t, dof>
        (num_diag, global_size_x / num_proc_x, global_size_y / num_proc_y, global_size_z / num_proc_z, 1, 1, 1);
    
    setup_comm_pkg();// 3d7的时候不需要角上的数据通信
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
par_structMatrix<idx_t, data_t, calc_t, dof>::par_structMatrix(const par_structMatrix & model) 
    : Operator<idx_t, data_t, calc_t>(  model.input_dim[0], model.input_dim[1], model.input_dim[2], 
                                        model.output_dim[0], model.output_dim[1], model.output_dim[2]),
      offset_x(model.offset_x), offset_y(model.offset_y), offset_z(model.offset_z)
{
    local_matrix = new seq_structMatrix<idx_t, data_t, calc_t, dof>(*(model.local_matrix));
    // 浅拷贝
    comm_pkg = model.comm_pkg;
    own_comm_pkg = false;
    stencil = model.stencil;
    own_sqrt_D = false;
    scaled = model.scaled;
    sqrt_D = model.sqrt_D;

    copy_irrgPts(model);
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::copy_irrgPts(const par_structMatrix & src)
{
    if (src.num_irrgPts > 0) {
        num_irrgPts = src.num_irrgPts;
        irrgPts = new irrgPts_mat<idx_t> [num_irrgPts];
        for (idx_t ir = 0; ir < num_irrgPts; ir++) 
            irrgPts[ir] = src.irrgPts[ir];

        const idx_t tot_nnz = irrgPts[num_irrgPts-1].beg + irrgPts[num_irrgPts-1].nnz;
        irrgPts_ngb_ijk = new idx_t [tot_nnz * 3];// i j k
        for (idx_t j = 0; j < tot_nnz * 3; j++)
            irrgPts_ngb_ijk[j] = src.irrgPts_ngb_ijk[j];

        irrgPts_A_vals = new data_t [tot_nnz * 2 * dof*dof];// two-sided effect
        for (idx_t j = 0; j < tot_nnz * 2 * dof*dof; j++)
            irrgPts_A_vals[j] = src.irrgPts_A_vals[j];
    }
    else {
        if (num_irrgPts > 0) {
            delete irrgPts; irrgPts = nullptr;
            delete irrgPts_ngb_ijk; irrgPts_ngb_ijk = nullptr;
            delete irrgPts_A_vals; irrgPts_A_vals = nullptr;
        }
        num_irrgPts = 0;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
par_structMatrix<idx_t, data_t, calc_t, dof>::~par_structMatrix()
{
    delete local_matrix;
    local_matrix = nullptr;
    if (own_comm_pkg) {
        delete comm_pkg;
        comm_pkg = nullptr;
    }
    if (LU_compressed) {
        delete Diag; Diag = nullptr;
        delete L_cprs; L_cprs = nullptr;
        delete U_cprs; U_cprs = nullptr;
    }
    if (scaled) {
        assert(sqrt_D != nullptr);
        delete sqrt_D; sqrt_D = nullptr;
    }
    if (num_irrgPts > 0) {
        delete irrgPts; irrgPts = nullptr;
        delete irrgPts_ngb_ijk; irrgPts_ngb_ijk = nullptr;
        delete irrgPts_A_vals; irrgPts_A_vals = nullptr;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::setup_cart_comm(MPI_Comm comm, 
    idx_t num_proc_x, idx_t num_proc_y, idx_t num_proc_z, bool unblk)
{
    // bool relay_mode = unblk ? false : true;
    bool relay_mode = true;
    comm_pkg = new StructCommPackage(relay_mode);
    own_comm_pkg = true;
    // 对comm_pkg内变量的引用，免得写太麻烦了
    MPI_Comm & cart_comm                         = comm_pkg->cart_comm;
    int (&cart_ids)[3]                           = comm_pkg->cart_ids;
    int (&ngbs_pid)[NUM_NEIGHBORS]               = comm_pkg->ngbs_pid;
    int & my_pid                                 = comm_pkg->my_pid;

    // create 2D distributed grid
    int dims[3] = {num_proc_y, num_proc_x, num_proc_z};
    int periods[3] = {0, 0, 0};

    MPI_Cart_create(comm, 3, dims, periods, 0, &cart_comm);
    assert(cart_comm != MPI_COMM_NULL);

    MPI_Cart_shift(cart_comm, 0, 1, &ngbs_pid[J_L], &ngbs_pid[J_U]);
    MPI_Cart_shift(cart_comm, 1, 1, &ngbs_pid[I_L], &ngbs_pid[I_U]);
    MPI_Cart_shift(cart_comm, 2, 1, &ngbs_pid[K_L], &ngbs_pid[K_U]);

    MPI_Comm_rank(cart_comm, &my_pid);
    MPI_Cart_coords(cart_comm, my_pid, 3, cart_ids);

#ifdef DEBUG
    printf("proc %3d cart_ids (%3d,%3d,%3d) IL %3d IU %3d JL %3d JU %3d KL %3d KU %3d\n",
        my_pid, cart_ids[0], cart_ids[1], cart_ids[2],  
        ngbs_pid[I_L], ngbs_pid[I_U], ngbs_pid[J_L], ngbs_pid[J_U], ngbs_pid[K_L], ngbs_pid[K_U]);
#endif
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::setup_comm_pkg(bool need_corner)
{
    MPI_Datatype (&send_subarray)[NUM_NEIGHBORS] = comm_pkg->send_subarray;
    MPI_Datatype (&recv_subarray)[NUM_NEIGHBORS] = comm_pkg->recv_subarray;
    MPI_Datatype & mpi_scalar_type               = comm_pkg->mpi_scalar_type;
    // 建立通信结构：注意data的排布从内到外依次为diag(3)->k(2)->i(1)->j(0)，按照C-order
    if     (sizeof(data_t) == 16)   mpi_scalar_type = MPI_LONG_DOUBLE;
    else if (sizeof(data_t) == 8)   mpi_scalar_type = MPI_DOUBLE;
    else if (sizeof(data_t) == 4)   mpi_scalar_type = MPI_FLOAT;
    else if (sizeof(data_t) == 2)   mpi_scalar_type = MPI_SHORT;
    else { printf("INVALID data_t when creating subarray, sizeof %ld bytes\n", sizeof(data_t)); MPI_Abort(MPI_COMM_WORLD, -2001); }

    idx_t size[5] = {   local_matrix->local_y + 2 * local_matrix->halo_y,
                        local_matrix->local_x + 2 * local_matrix->halo_x,
                        local_matrix->local_z + 2 * local_matrix->halo_z,
                        local_matrix->num_diag,
                        local_matrix->elem_size};
    idx_t subsize[5], send_start[5], recv_start[5];
    for (idx_t ingb = 0; ingb < NUM_NEIGHBORS; ingb++) {
        switch (ingb)
        {
        // 最先传的
        case K_L:
        case K_U:
            subsize[0] = local_matrix->local_y;
            subsize[1] = local_matrix->local_x;
            subsize[2] = local_matrix->halo_z;
            break;
        case I_L:
        case I_U:
            subsize[0] = local_matrix->local_y;
            subsize[1] = local_matrix->halo_x;
            subsize[2] = local_matrix->local_z + (need_corner ? 2 * local_matrix->halo_z : 0);
            break;
        case J_L:
        case J_U:
            subsize[0] = local_matrix->halo_y;
            subsize[1] = local_matrix->local_x + (need_corner ? 2 * local_matrix->halo_x : 0);
            subsize[2] = local_matrix->local_z + (need_corner ? 2 * local_matrix->halo_z : 0);
            break;
        default:
            printf("INVALID NEIGHBOR ID of %d!\n", ingb);
            MPI_Abort(MPI_COMM_WORLD, -2000);
        }
        // 最内维的高度层和矩阵元素层的通信长度不变
        subsize[3] = local_matrix->num_diag;
        subsize[4] = local_matrix->elem_size;

        switch (ingb)
        {
        case K_L:// 向K下发的内halo
            send_start[0] = recv_start[0] = local_matrix->halo_y;
            send_start[1] = recv_start[1] = local_matrix->halo_x;
            send_start[2] = local_matrix->halo_z;           recv_start[2] = 0;
            break;
        case K_U:
            send_start[0] = recv_start[0] = local_matrix->halo_y;
            send_start[1] = recv_start[1] = local_matrix->halo_x;
            send_start[2] = local_matrix->local_z;          recv_start[2] = local_matrix->local_z + local_matrix->halo_z;
            break;
        case I_L:
            send_start[0] = recv_start[0] = local_matrix->halo_y;
            send_start[1] = local_matrix->halo_x;           recv_start[1] = 0;
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_matrix->halo_z);
            break;
        case I_U:
            send_start[0] = recv_start[0] = local_matrix->halo_y;
            send_start[1] = local_matrix->local_x;          recv_start[1] = local_matrix->local_x + local_matrix->halo_x;
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_matrix->halo_z);
            break;
        case J_L:
            send_start[0] = local_matrix->halo_y;           recv_start[0] = 0;
            send_start[1] = recv_start[1] = (need_corner ? 0 : local_matrix->halo_x);
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_matrix->halo_z);
            break;
        case J_U:
            send_start[0] = local_matrix->local_y;          recv_start[0] = local_matrix->local_y + local_matrix->halo_y;
            send_start[1] = recv_start[1] = (need_corner ? 0 : local_matrix->halo_x);
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_matrix->halo_z);
            break;
        default:
            printf("INVALID NEIGHBOR ID of %d!\n", ingb);
            MPI_Abort(MPI_COMM_WORLD, -2000);
        }
        // 最内维的高度层的通信起始位置不变
        send_start[3] = recv_start[3] = 0;
        send_start[4] = recv_start[4] = 0;

        MPI_Type_create_subarray(5, size, subsize, send_start, MPI_ORDER_C, mpi_scalar_type, &send_subarray[ingb]);
        MPI_Type_commit(&send_subarray[ingb]);
        MPI_Type_create_subarray(5, size, subsize, recv_start, MPI_ORDER_C, mpi_scalar_type, &recv_subarray[ingb]);
        MPI_Type_commit(&recv_subarray[ingb]);
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::update_halo()
{
#ifdef DEBUG
    local_matrix->init_debug(offset_x, offset_y, offset_z);
    if (my_pid == 1) {
        local_matrix->print_level_diag(1, 3);
    }
#endif

    comm_pkg->exec_comm(local_matrix->data);

#ifdef DEBUG
    if (my_pid == 1) {
        local_matrix->print_level_diag(1, 3);
    }
#endif
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::init_irrPts(const std::string pathname)
{
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);

    std::string filename = pathname + "/irgP_a";
    FILE * fp = fopen(filename.c_str(), "r");
    idx_t gid, nnz;
    const idx_t lx = local_matrix->local_x, ly = local_matrix->local_y, lz = local_matrix->local_z;
    std::vector<idx_t> irrGid, irrNNZ;
    std::vector<std::tuple<idx_t,idx_t,idx_t> > irrNgbIJK;
    std::vector<std::pair<std::vector<data_t>, std::vector<data_t> > > irrNgbVal;
    while (fscanf(fp, "%d %d", &gid, &nnz) != EOF) {
        idx_t lie_in_my_domain = 0;// 0 for undetermined, 1 for yes, -1 for no
        idx_t i, j, k;
        std::vector<data_t> my_row(dof*dof, 0), other_row(dof*dof, 0);
        for (idx_t m = 0; m < nnz; m++) {
            fscanf(fp, "%d %d %d", &i, &j, &k);
            for (idx_t p = 0; p < dof*dof; p++)
                fscanf(fp, "%lf", &my_row[p]);
            for (idx_t p = 0; p < dof*dof; p++)
                fscanf(fp, "%lf", &other_row[p]);

            if (i == -1) {
                assert(j == -1 && k == -1);
                assert(lie_in_my_domain != 0);// 此前一定已经确定过了是否落在本进程范围内
            } else {
                if (lie_in_my_domain == 1) {
                    // printf("%d %d %d", i, j, k);
                    assert(offset_x <= i && i < offset_x + lx);
                    assert(offset_y <= j && j < offset_y + ly);
                    assert(offset_z <= k && k < offset_z + lz);
                }
                else if (lie_in_my_domain == 0) {
                    if (offset_x <= i && i < offset_x + lx && 
                        offset_y <= j && j < offset_y + ly &&
                        offset_z <= k && k < offset_z + lz   )
                        lie_in_my_domain =  1;// 落在本进程范围内
                    else
                        lie_in_my_domain = -1;
                }
            }

            if (lie_in_my_domain == -1) continue;

            irrNgbIJK.push_back(std::tuple<idx_t,idx_t,idx_t>(i, j, k));
            std::pair<std::vector<data_t>, std::vector<data_t> > tmp(my_row, other_row);
            irrNgbVal.push_back(tmp);
        }

        if (lie_in_my_domain == 1) {
            irrGid.push_back(gid);
            irrNNZ.push_back(nnz);
        }
    }
    fclose(fp);

    if (irrGid.size() > 0) {
        num_irrgPts = irrGid.size();
        irrgPts = new irrgPts_mat<idx_t> [num_irrgPts];
        for (idx_t ir = 0; ir < num_irrgPts; ir++) {
            irrgPts[ir].gid = irrGid[ir];
            irrgPts[ir].nnz = irrNNZ[ir];
            irrgPts[ir].beg = (ir == 0) ? 0 : (irrgPts[ir-1].beg + irrgPts[ir-1].nnz);
        }
        const idx_t tot_nnz = irrgPts[num_irrgPts-1].beg + irrgPts[num_irrgPts-1].nnz;
// #ifdef DEBUG
        printf(" proc %d got %ld irrgPts tot nnz %d\n", my_pid, irrGid.size(), tot_nnz);
// #endif
        irrgPts_ngb_ijk = new idx_t [tot_nnz * 3];// i j k
        irrgPts_A_vals = new data_t [tot_nnz * 2 * dof*dof];// two-sided effect
        for (idx_t p = 0; p < tot_nnz; p++) {
            irrgPts_ngb_ijk[p*3  ] = std::get<0>(irrNgbIJK[p]);
            irrgPts_ngb_ijk[p*3+1] = std::get<1>(irrNgbIJK[p]);
            irrgPts_ngb_ijk[p*3+2] = std::get<2>(irrNgbIJK[p]);
            
            for (idx_t f = 0; f < dof*dof; f++)
                irrgPts_A_vals [ p*2   *dof*dof + f] = irrNgbVal[p].first[f];
            for (idx_t f = 0; f < dof*dof; f++)
                irrgPts_A_vals [(p*2+1)*dof*dof + f] = irrNgbVal[p].second[f];
#ifdef DEBUG
            printf("    (%d,%d,%d) %.7e %.7e\n", 
                irrgPts_ngb_ijk[p*3  ], irrgPts_ngb_ijk[p*3+1], irrgPts_ngb_ijk[p*3+2],
                irrgPts_A_vals [p*2  ], irrgPts_A_vals [p*2+1]);
#endif
        }

        // 假定非规则点的对角元在最后一个
        for (idx_t ir = 0; ir < num_irrgPts; ir++) {
            idx_t last = irrgPts[ir].beg + irrgPts[ir].nnz - 1;
            assert(irrgPts_ngb_ijk[last*3  ] == -1 && irrgPts_ngb_ijk[last*3+1] == -1
                && irrgPts_ngb_ijk[last*3+2] == -1);
        }
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::Mult(const par_structVector<idx_t, calc_t, dof> & x,
                                                            par_structVector<idx_t, calc_t, dof> & y) const
{
    assert( this->input_dim[0] == x.global_size_x && this->output_dim[0] == y.global_size_x &&
            this->input_dim[1] == x.global_size_y && this->output_dim[1] == y.global_size_y &&
            this->input_dim[2] == x.global_size_z && this->output_dim[2] == y.global_size_z    );

    // lazy halo updated: only done when needed 
    x.update_halo();
#ifdef PROFILE
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    int num_procs; MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int num = LU_compressed ? (1 * local_matrix->elem_size + (num_diag - 1) * dof) : (num_diag * local_matrix->elem_size);
    double bytes = (local_matrix->local_x) * (local_matrix->local_y)
              * (local_matrix->local_z) * num * sizeof(data_t);
    int num_vec =  2 + int(scaled);
    bytes += (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
           * (x.local_vector->local_z + x.local_vector->halo_z * 2) * num_vec * dof * sizeof(calc_t);
    bytes *= num_procs;
    bytes /= (1024 * 1024 * 1024);// GB
    MPI_Barrier(y.comm_pkg->cart_comm);
    int warm_cnt = 3;
    for (int te = 0; te < warm_cnt; te++) {
        if (LU_compressed)
            compress_Mult(*(x.local_vector), *(y.local_vector));
        else
            local_matrix->Mult(*(x.local_vector), *(y.local_vector), sqrt_D);
    }
    MPI_Barrier(y.comm_pkg->cart_comm);
    double t = wall_time();
    int test_cnt = 5;
    for (int te = 0; te < test_cnt; te++) {
#endif
    // do computation
    if (LU_compressed)
        compress_Mult(*(x.local_vector), *(y.local_vector));
    else
        local_matrix->Mult(*(x.local_vector), *(y.local_vector), sqrt_D);
#ifdef PROFILE
    }
    t = wall_time() - t;
    t /= test_cnt;
    double mint, maxt;
    MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, y.comm_pkg->cart_comm);
    MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, y.comm_pkg->cart_comm);
    // mint = maxt = t;
    if (my_pid == 0) printf("SpMv data %ld calc %ld d%dB%dv%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                sizeof(data_t), sizeof(calc_t), num_diag, num, num_vec, bytes, mint, maxt, bytes/maxt, bytes/mint);
#endif

    // 非规则点的贡献// 前面的Mult()已经保证了三者的halo和local的宽度一致
    const idx_t hx = local_matrix->halo_x , hy = local_matrix->halo_y , hz = local_matrix->halo_z ;
    const idx_t lx = local_matrix->local_x, ly = local_matrix->local_y, lz = local_matrix->local_z;
    const idx_t vec_dki_size = x.local_vector->slice_dki_size, vec_dk_size = x.local_vector->slice_dk_size;
    assert(x.num_irrgPts == y.num_irrgPts && x.num_irrgPts == this->num_irrgPts);
    const idx_t blk_size = dof*dof;
    for (idx_t ir = 0; ir < num_irrgPts; ir++) {
        assert(x.irrgPts[ir].gid == y.irrgPts[ir].gid && x.irrgPts[ir].gid == this->irrgPts[ir].gid);
        const idx_t pbeg = irrgPts[ir].beg, pend = pbeg + irrgPts[ir].nnz;
        
        calc_t * res = y.irrgPts[ir].val;
        vec_zero<idx_t, calc_t, dof>(res);
        for (idx_t p = pbeg; p < pend; p++) {// 遍历本非结构点的所有邻居
            const idx_t ngb_i = irrgPts_ngb_ijk[p*3  ],
                        ngb_j = irrgPts_ngb_ijk[p*3+1],
                        ngb_k = irrgPts_ngb_ijk[p*3+2];// global coord
            if (p == pend - 1) {// 对角元，受自己这个非规则点的影响
                assert(ngb_i == -1 && ngb_j == -1 && ngb_k == -1);
                // res += irrgPts_A_vals[p*2] * x.irrgPts[ir].val;
                matvec_mla<idx_t, data_t, calc_t, dof>(irrgPts_A_vals + (p<<1)*blk_size, x.irrgPts[ir].val, res);
            } else {
                assert(offset_x <= ngb_i && ngb_i < offset_x + lx);
                assert(offset_y <= ngb_j && ngb_j < offset_y + ly);
                assert(offset_z <= ngb_k && ngb_k < offset_z + lz);
                const idx_t loc_i = hx + ngb_i - offset_x,
                            loc_j = hy + ngb_j - offset_y,
                            loc_k = hz + ngb_k - offset_z;
                idx_t loc_1D = dof * loc_k + vec_dk_size * loc_i + vec_dki_size * loc_j;
                // 本非规则点受其它结构点的影响
                // res += irrgPts_A_vals[p*2] * x.local_vector->data[loc_1D];
                matvec_mla<idx_t, data_t, calc_t, dof>(irrgPts_A_vals + (p<<1)*blk_size, x.local_vector->data + loc_1D, res);
                // 其它结构点受本非规则点的影响
                // y.local_vector->data[loc_1D] += irrgPts_A_vals[p*2 + 1] * x.irrgPts[ir].val;
                matvec_mla<idx_t, data_t, calc_t, dof>(irrgPts_A_vals + ((p<<1)+1)*blk_size, x.irrgPts[ir].val, y.local_vector->data + loc_1D);
            }
        }
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::read_data(const std::string pathname) {
    seq_structMatrix<idx_t, data_t, calc_t, dof> & A_local = *local_matrix;

    assert( (sizeof(data_t) ==16 && comm_pkg->mpi_scalar_type == MPI_LONG_DOUBLE) || 
            (sizeof(data_t) == 4 && comm_pkg->mpi_scalar_type == MPI_FLOAT)  ||
            (sizeof(data_t) == 8 && comm_pkg->mpi_scalar_type == MPI_DOUBLE) ||
            (sizeof(data_t) == 2 && comm_pkg->mpi_scalar_type == MPI_SHORT )    );
    
    idx_t lx = A_local.local_x, ly = A_local.local_y, lz = A_local.local_z;
    idx_t lms = A_local.elem_size;
    idx_t tot_len = lx * ly * lz * lms;
    double * buf = new double [tot_len];// 读入缓冲区（存储的数据是32位的，只能按此格式读入）

    MPI_File fh = MPI_FILE_NULL;// 文件句柄
    MPI_Datatype etype = MPI_DOUBLE;// 给定的数据是双精度的
    MPI_Datatype read_type = MPI_DATATYPE_NULL;// 读取时的类型

    
    idx_t size[4], subsize[4], start[4];
    size[0] = this->input_dim[1];// y 方向的全局大小
    size[1] = this->input_dim[0];// x 方向的全局大小
    size[2] = this->input_dim[2];// z 方向的全局大小
    size[3] = lms;
    subsize[0] = ly;
    subsize[1] = lx;
    subsize[2] = lz;
    subsize[3] = size[3];
    start[0] = offset_y;
    start[1] = offset_x;
    start[2] = offset_z;
    start[3] = 0;
    
    MPI_Type_create_subarray(4, size, subsize, start, MPI_ORDER_C, etype, &read_type);
    MPI_Type_commit(&read_type);

    MPI_Offset displacement = 0;
    displacement *= sizeof(double);// 位移要以字节为单位！
    MPI_Status status;

    // 依次读入A的各条对角线
    for (idx_t idiag = 0; idiag < num_diag; idiag++) {
        // 注意对角线数据的存储文件命名是从1开始的
        MPI_File_open(comm_pkg->cart_comm, (pathname + "/array_a." + std::to_string(idiag)).c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        MPI_File_set_view(fh, displacement, etype, read_type, "native", MPI_INFO_NULL);
        MPI_File_read_all(fh, buf, tot_len, etype, &status);
        // （可能存在截断）转入A矩阵内
        for (idx_t j = 0; j < ly; j++)
        for (idx_t i = 0; i < lx; i++)
        for (idx_t k = 0; k < lz; k++)
        for (idx_t e = 0; e < lms; e++)
            A_local.data[e + idiag * lms +
                        (k + A_local.halo_z) * A_local.slice_ed_size + 
                        (i + A_local.halo_x) * A_local.slice_edk_size +
                        (j + A_local.halo_y) * A_local.slice_edki_size]
                    = (data_t) buf[e + lms * (k + lz * (i + lx * j))];
        MPI_File_close(&fh);
    }
    // 矩阵需要填充halo区（目前仅此一次）
    update_halo();

    delete buf;
    buf = nullptr;
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::set_val(data_t val, bool halo_set) {
    if (halo_set) {
        const idx_t tot_len = (local_matrix->local_x + local_matrix->halo_x * 2)
                            * (local_matrix->local_y + local_matrix->halo_y * 2)
                            * (local_matrix->local_z + local_matrix->halo_z * 2)
                            *  local_matrix->num_diag * local_matrix->elem_size;
        for (idx_t p = 0; p < tot_len; p++)
            local_matrix->data[p] = 0.0;
    }
    else
        *(local_matrix) = val;
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::set_diag_val(idx_t d, data_t val) {
    local_matrix->set_diag_val(d, val);
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::compress_LU() const
{
    int my_pid; MPI_Comm_rank(comm_pkg->cart_comm, &my_pid);
    if (LU_compressed) {// 若之前已经分离过了，重新分离
        if (my_pid == 0) printf("Compressed LU but do compression again\n");
        delete Diag;
        delete L_cprs; delete U_cprs;
    } else {
        if (my_pid == 0) printf("Compress LU\n");
    }

    idx_t lms = local_matrix->elem_size;
    assert(lms == dof*dof);

    const idx_t lx = local_matrix->local_x, ly = local_matrix->local_y, lz = local_matrix->local_z,
                hx = local_matrix->halo_x , hy = local_matrix->halo_y , hz = local_matrix->halo_z ;
    const idx_t diag_id = num_diag >> 1;
    Diag = new seq_structVector<idx_t, data_t, dof*dof>(lx, ly, lz, hx, hy, hz);
    L_cprs = new seq_structMatrix<idx_t, data_t, calc_t, 1>(diag_id * dof, lx, ly, lz, hx, hy, hz);
    U_cprs = new seq_structMatrix<idx_t, data_t, calc_t, 1>(*L_cprs);

    idx_t tot_cells= (lx + hx * 2) * (ly + hy * 2) * (lz + hz * 2);
    #pragma omp parallel for schedule(static)
    for (idx_t ic = 0; ic < tot_cells; ic++) {// 逐个单元
        const data_t * cell_data = local_matrix->data + ic * local_matrix->slice_ed_size;

        data_t * L_ptr = L_cprs->data + ic * L_cprs->slice_ed_size;// 因为压缩后的L和U自由度填了1，所以elem_size=1，ed_size就是对角线
        for (idx_t id = 0; id < diag_id; id++) {// 压缩L
            for (idx_t f = 0; f < dof; f++)
                L_ptr[id * dof + f] = cell_data[id * lms + f * dof + f];
        }
        // 对角块不压缩
        data_t * D_ptr = Diag->data + ic * lms;
        for (idx_t f = 0; f < lms; f++)
            D_ptr[f] = cell_data[diag_id * lms + f];

        data_t * U_ptr = U_cprs->data + ic * U_cprs->slice_ed_size;
        for (idx_t id = diag_id + 1; id < num_diag; id++) {// 压缩U
            for (idx_t f = 0; f < dof; f++)
                U_ptr[(id - diag_id - 1) * dof + f] = cell_data[id * lms + f * dof + f];
        }
    }
    LU_compressed = true;
}


template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::compress_Mult(const seq_structVector<idx_t, calc_t, dof> & x,
                                                                    seq_structVector<idx_t, calc_t, dof> & y) const
{
    assert(LU_compressed);
    CHECK_LOCAL_HALO(*local_matrix, x);
    CHECK_LOCAL_HALO(x , y);
    const calc_t * x_data = x.data;
    calc_t * y_data = y.data;
    const data_t * L_data = L_cprs->data, * U_data = U_cprs->data, * D_data = Diag->data;
    const calc_t * sqD_data = scaled ? sqrt_D->data : nullptr;
    const idx_t lms = dof*dof;

    void (*kernel)(const idx_t, const idx_t, const idx_t, const data_t*, const data_t*, const data_t*,
        const calc_t*, calc_t*, const calc_t*) = scaled ? compress_spmv_scaled : compress_spmv;
    assert(kernel);
    
    const idx_t ibeg = local_matrix->halo_x, iend = ibeg + local_matrix->local_x,
                jbeg = local_matrix->halo_y, jend = jbeg + local_matrix->local_y,
                kbeg = local_matrix->halo_z, kend = kbeg + local_matrix->local_z;
    const idx_t vec_dk_size = x.slice_dk_size, vec_dki_size = x.slice_dki_size;
    const idx_t LU_edki_size = L_cprs->slice_edki_size, LU_edk_size = L_cprs->slice_edk_size, LU_ed_size = L_cprs->slice_ed_size;
    const idx_t col_height = kend - kbeg;

    #pragma omp parallel for collapse(2) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        const idx_t LU_off = j * LU_edki_size + i * LU_edk_size + kbeg * LU_ed_size;
        const data_t * L_jik = L_data + LU_off, * U_jik = U_data + LU_off;
        const data_t * D_jik = D_data + j * Diag->slice_dki_size + i * Diag->slice_dk_size + kbeg * lms;
        const idx_t vec_off = j * vec_dki_size + i * vec_dk_size + kbeg * dof;
        const calc_t * sqD_jik = (sqD_data) ? (sqD_data + vec_off) : nullptr;
        const calc_t * x_jik = x_data + vec_off;
        calc_t * y_jik = y_data + vec_off;
        kernel(col_height, vec_dk_size, vec_dki_size, L_jik, D_jik, U_jik, x_jik, y_jik, sqD_jik);
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
bool par_structMatrix<idx_t, data_t, calc_t, dof>::check_Dirichlet()
{
    // 确定各维上是否是边界
    const bool x_lbdr = comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;

    const idx_t ibeg = local_matrix->halo_x, iend = ibeg + local_matrix->local_x,
                jbeg = local_matrix->halo_y, jend = jbeg + local_matrix->local_y,
                kbeg = local_matrix->halo_z, kend = kbeg + local_matrix->local_z;
    const idx_t lms = local_matrix->elem_size;
    std::unordered_set<idx_t> check_ids;

    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        check_ids.clear();

        const data_t * ptr = local_matrix->data + j * local_matrix->slice_edki_size + 
            i * local_matrix->slice_edk_size + k * local_matrix->slice_ed_size;
        if (num_diag == 27) {
            if (i == ibeg   && x_lbdr) {
                check_ids.insert(0); check_ids.insert(1); check_ids.insert(2);
                check_ids.insert(9); check_ids.insert(10);check_ids.insert(11);
                check_ids.insert(18);check_ids.insert(19);check_ids.insert(20);
            }
            if (i == iend-1 && x_ubdr) {
                check_ids.insert(6); check_ids.insert(7); check_ids.insert(8);
                check_ids.insert(15);check_ids.insert(16);check_ids.insert(17);
                check_ids.insert(24);check_ids.insert(25);check_ids.insert(26);
            }
            if (j == jbeg   && y_lbdr) {
                check_ids.insert(0); check_ids.insert(1); check_ids.insert(2);
                check_ids.insert(3); check_ids.insert(4); check_ids.insert(5);
                check_ids.insert(6); check_ids.insert(7); check_ids.insert(8);
            }
            if (j == jend-1 && y_ubdr) {
                check_ids.insert(18); check_ids.insert(19); check_ids.insert(20);
                check_ids.insert(21); check_ids.insert(22); check_ids.insert(23);
                check_ids.insert(24); check_ids.insert(25); check_ids.insert(26);
            }
            if (k == kbeg   && z_lbdr) {
                check_ids.insert(0); check_ids.insert(3); check_ids.insert(6);
                check_ids.insert(9); check_ids.insert(12);check_ids.insert(15);
                check_ids.insert(18);check_ids.insert(21);check_ids.insert(24);
            }
            if (k == kend-1 && z_ubdr) {
                check_ids.insert(2); check_ids.insert(5); check_ids.insert(8);
                check_ids.insert(11);check_ids.insert(14);check_ids.insert(17);
                check_ids.insert(20);check_ids.insert(23);check_ids.insert(26);
            }
        }
        else if (num_diag == 7) {
            if (j == jbeg   && y_lbdr) check_ids.insert(0);
            if (i == ibeg   && x_lbdr) check_ids.insert(1);
            if (k == kbeg   && z_lbdr) check_ids.insert(2);
            if (k == kend-1 && z_ubdr) check_ids.insert(4);
            if (i == iend-1 && x_ubdr) check_ids.insert(5);
            if (j == jend-1 && y_ubdr) check_ids.insert(6);
        }
        else if (num_diag == 15) {
            if (i == ibeg && x_lbdr) {
                check_ids.insert(0); check_ids.insert(1); check_ids.insert(4); check_ids.insert(5);
            }
            if (i == iend-1 && x_ubdr) {
                check_ids.insert(9); check_ids.insert(10);check_ids.insert(13);check_ids.insert(14);
            }
            if (j == jbeg   && y_lbdr) {
                check_ids.insert(0); check_ids.insert(1); check_ids.insert(2); check_ids.insert(3);
            }
            if (j == jend-1 && y_ubdr) {
                check_ids.insert(11);check_ids.insert(12);check_ids.insert(13);check_ids.insert(14);
            }
            if (k == kbeg   && z_lbdr) {
                check_ids.insert(0); check_ids.insert(2); check_ids.insert(4); check_ids.insert(6);
            }
            if (k == kend-1 && z_ubdr) {
                check_ids.insert(8); check_ids.insert(10);check_ids.insert(12);check_ids.insert(14);
            }
        }
        else {
            assert(false);
        }
        for (typename std::unordered_set<idx_t>::iterator it = check_ids.begin(); it != check_ids.end(); it++) {
            idx_t id = *it;
            const data_t * blk_ptr = ptr + id * lms;
            for (idx_t p = 0; p < lms; p++)
                assert(blk_ptr[p] == 0.0);
        }
    }

    return true;
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::set_boundary()
{
    // 确定各维上是否是边界
    const bool x_lbdr = comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;

    const idx_t ibeg = local_matrix->halo_x, iend = ibeg + local_matrix->local_x,
                jbeg = local_matrix->halo_y, jend = jbeg + local_matrix->local_y,
                kbeg = local_matrix->halo_z, kend = kbeg + local_matrix->local_z;
    const idx_t lms = local_matrix->elem_size;
    std::unordered_set<idx_t> check_ids;

    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        check_ids.clear();

        data_t * ptr = local_matrix->data + j * local_matrix->slice_edki_size + 
            i * local_matrix->slice_edk_size + k * local_matrix->slice_ed_size;
        if (num_diag == 27) {
            if (i == ibeg   && x_lbdr) {
                check_ids.insert(0); check_ids.insert(1); check_ids.insert(2);
                check_ids.insert(9); check_ids.insert(10);check_ids.insert(11);
                check_ids.insert(18);check_ids.insert(19);check_ids.insert(20);
            }
            if (i == iend-1 && x_ubdr) {
                check_ids.insert(6); check_ids.insert(7); check_ids.insert(8);
                check_ids.insert(15);check_ids.insert(16);check_ids.insert(17);
                check_ids.insert(24);check_ids.insert(25);check_ids.insert(26);
            }
            if (j == jbeg   && y_lbdr) {
                check_ids.insert(0); check_ids.insert(1); check_ids.insert(2);
                check_ids.insert(3); check_ids.insert(4); check_ids.insert(5);
                check_ids.insert(6); check_ids.insert(7); check_ids.insert(8);
            }
            if (j == jend-1 && y_ubdr) {
                check_ids.insert(18); check_ids.insert(19); check_ids.insert(20);
                check_ids.insert(21); check_ids.insert(22); check_ids.insert(23);
                check_ids.insert(24); check_ids.insert(25); check_ids.insert(26);
            }
            if (k == kbeg   && z_lbdr) {
                check_ids.insert(0); check_ids.insert(3); check_ids.insert(6);
                check_ids.insert(9); check_ids.insert(12);check_ids.insert(15);
                check_ids.insert(18);check_ids.insert(21);check_ids.insert(24);
            }
            if (k == kend-1 && z_ubdr) {
                check_ids.insert(2); check_ids.insert(5); check_ids.insert(8);
                check_ids.insert(11);check_ids.insert(14);check_ids.insert(17);
                check_ids.insert(20);check_ids.insert(23);check_ids.insert(26);
            }
        }
        else if (num_diag == 7) {
            if (j == jbeg   && y_lbdr) check_ids.insert(0);
            if (i == ibeg   && x_lbdr) check_ids.insert(1);
            if (k == kbeg   && z_lbdr) check_ids.insert(2);
            if (k == kend-1 && z_ubdr) check_ids.insert(4);
            if (i == iend-1 && x_ubdr) check_ids.insert(5);
            if (j == jend-1 && y_ubdr) check_ids.insert(6);
        }
        else if (num_diag == 15) {
            if (i == ibeg && x_lbdr) {
                check_ids.insert(0); check_ids.insert(1); check_ids.insert(4); check_ids.insert(5);
            }
            if (i == iend-1 && x_ubdr) {
                check_ids.insert(9); check_ids.insert(10);check_ids.insert(13);check_ids.insert(14);
            }
            if (j == jbeg   && y_lbdr) {
                check_ids.insert(0); check_ids.insert(1); check_ids.insert(2); check_ids.insert(3);
            }
            if (j == jend-1 && y_ubdr) {
                check_ids.insert(11);check_ids.insert(12);check_ids.insert(13);check_ids.insert(14);
            }
            if (k == kbeg   && z_lbdr) {
                check_ids.insert(0); check_ids.insert(2); check_ids.insert(4); check_ids.insert(6);
            }
            if (k == kend-1 && z_ubdr) {
                check_ids.insert(8); check_ids.insert(10);check_ids.insert(12);check_ids.insert(14);
            }
        }
        else {
            assert(false);
        }
        for (typename std::unordered_set<idx_t>::iterator it = check_ids.begin(); it != check_ids.end(); it++) {
            idx_t id = *it;
            data_t * blk_ptr = ptr + id * lms;
            for (idx_t p = 0; p < lms; p++)
                blk_ptr[p] = 0.0;
        }
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::scale(const data_t scaled_diag)
{
    assert(scaled == false);
    assert(sizeof(data_t) == sizeof(calc_t));
    sqrt_D = new seq_structVector<idx_t, calc_t, dof>(
        local_matrix->local_x, local_matrix->local_y, local_matrix->local_z,
        local_matrix->halo_x , local_matrix->halo_y , local_matrix->halo_z  );
    sqrt_D->set_halo(0.0);
    // 确定各维上是否是边界
    const bool x_lbdr = comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;

    idx_t jbeg = (y_lbdr ? sqrt_D->halo_y : 0), jend = sqrt_D->halo_y + sqrt_D->local_y + (y_ubdr ? 0 : sqrt_D->halo_y);
    idx_t ibeg = (x_lbdr ? sqrt_D->halo_x : 0), iend = sqrt_D->halo_x + sqrt_D->local_x + (x_ubdr ? 0 : sqrt_D->halo_x);
    idx_t kbeg = (z_lbdr ? sqrt_D->halo_z : 0), kend = sqrt_D->halo_z + sqrt_D->local_z + (z_ubdr ? 0 : sqrt_D->halo_z);

    int my_pid; MPI_Comm_rank(comm_pkg->cart_comm, &my_pid);
    if (my_pid == 0) printf("parMat scaled => diagonal as %.2e\n", scaled_diag);

    CHECK_LOCAL_HALO(*sqrt_D, *local_matrix);
    const idx_t vec_dki_size = sqrt_D->slice_dki_size, vec_dk_size = sqrt_D->slice_dk_size;
    const idx_t slice_edki_size = local_matrix->slice_edki_size, slice_edk_size = local_matrix->slice_edk_size,
                slice_ed_size = local_matrix->slice_ed_size, slice_e_size = local_matrix->elem_size;
    // 提取对角线元素，开方
    const idx_t diag_id = num_diag >> 1;
    #pragma omp parallel for collapse(3) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        const data_t * src_ptr = local_matrix->data + j * slice_edki_size + i * slice_edk_size + k * slice_ed_size + diag_id * slice_e_size;
        calc_t * dst_ptr = sqrt_D->data + j * vec_dki_size + i * vec_dk_size + k * dof;
        
        // 各自由度自己做
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++) {
            data_t tmp = src_ptr[f*dof + f];
            assert(tmp > 0.0);
            tmp /= scaled_diag;
            dst_ptr[f] = sqrt(tmp);
        }
        
        // // 计算（多个自由度合在一块的）方阵的行列式
        // data_t det = calc_det_3x3<idx_t, data_t, dof>(src_ptr);
        // det = pow(det, 1.0 / dof);
        // data_t tmp = sqrt(det / scaled_diag);
        // for (idx_t f = 0; f < dof; f++)
        //     dst_ptr[f] = tmp;
        // printf("(%d,%d,%d) %.5e %.5e %.5e\n", i, j, k, dst_ptr[0], dst_ptr[1], dst_ptr[2]);
    }
    // 矩阵元素的scaling
    jbeg = local_matrix->halo_y; jend = jbeg + local_matrix->local_y;
    ibeg = local_matrix->halo_x; iend = ibeg + local_matrix->local_x;
    kbeg = local_matrix->halo_z; kend = kbeg + local_matrix->local_z;
    #pragma omp parallel for collapse(3) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++)
    for (idx_t d = 0; d < num_diag; d++) {
        data_t * mat_ptr = local_matrix->data + j * slice_edki_size + i * slice_edk_size + k * slice_ed_size + d * slice_e_size;
        idx_t ngb_j = j + stencil[d*3 + 0];
        idx_t ngb_i = i + stencil[d*3 + 1];
        idx_t ngb_k = k + stencil[d*3 + 2];
        bool all_zero = true;
        for (idx_t e = 0; e < slice_e_size; e++)
            all_zero = all_zero && (mat_ptr[e] == 0.0);
        if      (x_lbdr && ngb_i <  ibeg) assert(all_zero);
        else if (x_ubdr && ngb_i >= iend) assert(all_zero);
        else if (y_lbdr && ngb_j <  jbeg) assert(all_zero);
        else if (y_ubdr && ngb_j >= jend) assert(all_zero);
        else if (z_lbdr && ngb_k <  kbeg) assert(all_zero);
        else if (z_ubdr && ngb_k >= kend) assert(all_zero);
        else {
            for (idx_t r = 0; r < dof; r++)
            for (idx_t c = 0; c < dof; c++) {
                data_t tmp = mat_ptr[r * dof + c];
                calc_t my_sqrt_Dval = sqrt_D->data[    j * vec_dki_size +     i * vec_dk_size +     k * dof + r];
                calc_t ngb_sqrt_Dval= sqrt_D->data[ngb_j * vec_dki_size + ngb_i * vec_dk_size + ngb_k * dof + c];
                assert(my_sqrt_Dval > 0.0 && ngb_sqrt_Dval > 0.0);
                tmp /= (my_sqrt_Dval * ngb_sqrt_Dval);
                // if (tmp - scaled_diag > 1e-4) {
                //     printf("(%d,%d,%d) dir %d r %d c %d old %.5e my %.5e ngb %.5e tmp %.5e\n", 
                //         i, j, k, d, r, c, mat_ptr[r * dof + c], my_sqrt_Dval, ngb_sqrt_Dval, tmp);
                // }
                mat_ptr[r * dof + c] = tmp;
            }
        }
    }
    update_halo();// 更新一下scaling之后的矩阵元素
    assert(check_Dirichlet());
    assert(check_scaling(scaled_diag));
    scaled = true;

    if (LU_compressed) compress_LU();// 需要重新压缩非对角部分
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
bool par_structMatrix<idx_t, data_t, calc_t, dof>::check_scaling(const data_t scaled_diag)
{
    const idx_t ibeg = local_matrix->halo_x, iend = ibeg + local_matrix->local_x,
                jbeg = local_matrix->halo_y, jend = jbeg + local_matrix->local_y,
                kbeg = local_matrix->halo_z, kend = kbeg + local_matrix->local_z;
    idx_t id = -1;
    if (num_diag == 27)
        id = 13;
    else if (num_diag == 7)
        id = 3;
    assert(id != -1);
    #pragma omp parallel for collapse(3) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        const data_t * ptr = local_matrix->data + j * local_matrix->slice_edki_size + 
            i * local_matrix->slice_edk_size + k * local_matrix->slice_ed_size + id * local_matrix->elem_size;
        
        for (idx_t f = 0; f < dof; f++)
            assert(abs(ptr[f*dof + f] - scaled_diag) < 1e-4);

        // data_t det = calc_det_3x3<idx_t, data_t, dof>(ptr);
        // if (abs(det - scaled_diag) >= 1e-4) {
        //     printf(" (%d,%d,%d)\n", i, j, k);
        //         for (idx_t e = 0; e < local_matrix->elem_size; e += dof)
        //             printf("  %.4e %.4e %.4e\n", ptr[e], ptr[e+1], ptr[e+2]);
        //         printf("  det %.8e scaled_diag %.8e abs(diff) %.8e\n", det, scaled_diag, abs(det - scaled_diag));
        // }
        // assert(abs(det - scaled_diag) < 1e-4);
        
        // for (idx_t e = 0; e < local_matrix->elem_size; e += dof) {
        //     if (!(abs(ptr[e] - scaled_diag) < 1e-4)) {
        //         printf(" (%d,%d,%d)\n", i, j, k);
        //         for (idx_t e = 0; e < local_matrix->elem_size; e += dof)
        //             printf("  %.4e %.4e %.4e\n", ptr[e], ptr[e+1], ptr[e+2]);
        //         printf("  ptr[e] %.8e scaled_diag %.8e abs(diff) %.8e\n", ptr[e], scaled_diag, abs(ptr[e] - scaled_diag));
        //     }
        //     assert(abs(ptr[e] - scaled_diag) < 1e-4);
        // }
    }
    return true;
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::write_CSR_bin() const {
    int num_proc; MPI_Comm_size(comm_pkg->cart_comm, &num_proc);
    assert(num_proc == 1);
    assert(sizeof(data_t) == 8);// double only
    assert(sizeof(idx_t)  == 4);// int only
    assert(this->input_dim[0] == this->output_dim[0] && this->input_dim[1] == this->output_dim[1] && this->input_dim[2] == this->output_dim[2]);

    const idx_t gx = this->input_dim[0], gy = this->input_dim[1], gz = this->input_dim[2];
    const long long nrows = gx * gy * gz * dof;

    if (nrows * 45 < 2147483647) {// 非零元仍然可以用整型数表示
        std::vector<int> row_ptr(nrows+1, 0);
        std::vector<int> col_idx;
        std::vector<data_t> vals;
        const idx_t jbeg = local_matrix->halo_y, jend = jbeg + local_matrix->local_y,
                    ibeg = local_matrix->halo_x, iend = ibeg + local_matrix->local_x,
                    kbeg = local_matrix->halo_z, kend = kbeg + local_matrix->local_z;
        for (int j = jbeg; j < jend; j++)
        for (int i = ibeg; i < iend; i++)
        for (int k = kbeg; k < kend; k++)
        for (int f = 0; f < dof; f++) {// 一个自由度就是一行
            int row_idx = (((j-jbeg)*gx + i-ibeg)*gz + k-kbeg)*dof + f; assert(row_idx < nrows);
            for (int d = 0; d < num_diag; d++) {
                const int   ngb_j = j + stencil_offset_3d15[d*3  ],
                            ngb_i = i + stencil_offset_3d15[d*3+1],
                            ngb_k = k + stencil_offset_3d15[d*3+2];
                if (ngb_j < jbeg || ngb_j >= jend || 
                    ngb_i < ibeg || ngb_i >= iend || 
                    ngb_k < kbeg || ngb_k >= kend   ) continue;
                for (int ngb_f = 0; ngb_f < dof; ngb_f++) {
                    int ngb_row_idx = (((ngb_j-jbeg)*gx + ngb_i-ibeg)*gz + ngb_k-kbeg)*dof + ngb_f; assert(ngb_row_idx < nrows);
                    data_t val = local_matrix->data[j * local_matrix->slice_edki_size + i * local_matrix->slice_edk_size
                        + k * local_matrix->slice_ed_size + d * local_matrix->elem_size + f*dof + ngb_f];
                    if (val != 0.0) {// 非零元：剔除掉了零元素！！！
                        row_ptr[row_idx+1] ++;
                        col_idx.push_back(ngb_row_idx);
                        vals.push_back(val);
                    }
                }
            }// d
        }// f
        
        for (idx_t i = 0; i < nrows; i++)
            row_ptr[i+1] += row_ptr[i];
        assert(row_ptr[0] == 0);
        assert(row_ptr[nrows] == col_idx.size());
        assert(row_ptr[nrows] == vals.size());
        printf("tot_nnz %d\n", row_ptr[nrows]);

        FILE * fp = nullptr; int size;
        fp = fopen("Ai.bin", "wb");
        size = fwrite(row_ptr.data(), sizeof(int)   , nrows + 1     , fp); assert(size == nrows + 1     ); fclose(fp);
        fp = fopen("Aj.bin", "wb");
        size = fwrite(col_idx.data(), sizeof(int)   , row_ptr[nrows], fp); assert(size == row_ptr[nrows]); fclose(fp);
        fp = fopen("Av.bin", "wb");
        size = fwrite(vals.data()   , sizeof(data_t), row_ptr[nrows], fp); assert(size == row_ptr[nrows]); fclose(fp);
    }
    else {
        printf("using BigInt as row_ptr and col_idx\n");
        std::vector<long long> row_ptr(nrows+1, 0);
        std::vector<long long> col_idx;
        std::vector<data_t> vals;
        const idx_t jbeg = local_matrix->halo_y, jend = jbeg + local_matrix->local_y,
                    ibeg = local_matrix->halo_x, iend = ibeg + local_matrix->local_x,
                    kbeg = local_matrix->halo_z, kend = kbeg + local_matrix->local_z;
        for (int j = jbeg; j < jend; j++)
        for (int i = ibeg; i < iend; i++)
        for (int k = kbeg; k < kend; k++)
        for (int f = 0; f < dof; f++) {// 一个自由度就是一行
            long long row_idx = (((j-jbeg)*gx + i-ibeg)*gz + k-kbeg)*dof + f; assert(row_idx < nrows);
            for (int d = 0; d < num_diag; d++) {
                const int   ngb_j = j + stencil_offset_3d15[d*3  ],
                            ngb_i = i + stencil_offset_3d15[d*3+1],
                            ngb_k = k + stencil_offset_3d15[d*3+2];
                if (ngb_j < jbeg || ngb_j >= jend || 
                    ngb_i < ibeg || ngb_i >= iend || 
                    ngb_k < kbeg || ngb_k >= kend   ) continue;
                for (int ngb_f = 0; ngb_f < dof; ngb_f++) {
                    long long ngb_row_idx = (((ngb_j-jbeg)*gx + ngb_i-ibeg)*gz + ngb_k-kbeg)*dof + ngb_f; assert(ngb_row_idx < nrows);
                    data_t val = local_matrix->data[j * local_matrix->slice_edki_size + i * local_matrix->slice_edk_size
                        + k * local_matrix->slice_ed_size + d * local_matrix->elem_size + f*dof + ngb_f];
                    if (val != 0.0) {// 非零元：剔除掉了零元素！！！
                        row_ptr[row_idx+1] ++;
                        col_idx.push_back(ngb_row_idx);
                        vals.push_back(val);
                    }
                }
            }// d
        }// f
        
        for (idx_t i = 0; i < nrows; i++)
            row_ptr[i+1] += row_ptr[i];
        assert(row_ptr[0] == 0);
        assert(row_ptr[nrows] == col_idx.size());
        assert(row_ptr[nrows] == vals.size());

        FILE * fp = nullptr; long long size;
        fp = fopen("Ai.bin", "wb");
        size = fwrite(row_ptr.data(), sizeof(long long), nrows + 1     , fp); assert(size == nrows + 1     ); fclose(fp);
        fp = fopen("Aj.bin", "wb");
        size = fwrite(col_idx.data(), sizeof(long long), row_ptr[nrows], fp); assert(size == row_ptr[nrows]); fclose(fp);
        fp = fopen("Av.bin", "wb");
        size = fwrite(vals.data()   , sizeof(data_t)   , row_ptr[nrows], fp); assert(size == row_ptr[nrows]); fclose(fp);
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void par_structMatrix<idx_t, data_t, calc_t, dof>::write_struct_AOS_bin(const std::string pathname, const std::string file)
{
    int my_pid; MPI_Comm_rank(comm_pkg->cart_comm, &my_pid);
    seq_structMatrix<idx_t, data_t, calc_t, dof> & mat = *local_matrix;

    idx_t lx = mat.local_x, ly = mat.local_y, lz = mat.local_z;
    const idx_t nnz_len = num_diag * mat.elem_size;

    idx_t tot_len = lx * ly * lz * nnz_len;
    assert(sizeof(data_t) == sizeof(double));
    double * buf = new double[tot_len];// 读入缓冲区（给定的数据是双精度的）
    MPI_File fh = MPI_FILE_NULL;// 文件句柄
    MPI_Datatype etype = MPI_DOUBLE;// 给定的数据是双精度的
    MPI_Datatype write_type = MPI_DATATYPE_NULL;// 读取时的类型

    
    idx_t size[4], subsize[4], start[4];
    size[0] = this->input_dim[1];// global_size_y;
    size[1] = this->input_dim[0];// global_size_x;
    size[2] = this->input_dim[2];// global_size_z;
    size[3] = nnz_len;
    subsize[0] = ly;
    subsize[1] = lx;
    subsize[2] = lz;
    subsize[3] = nnz_len;
    start[0] = offset_y;
    start[1] = offset_x;
    start[2] = offset_z;
    start[3] = 0;

    MPI_Type_create_subarray(4, size, subsize, start, MPI_ORDER_C, etype, &write_type);
    MPI_Type_commit(&write_type);

    MPI_Offset displacement = 0;
    displacement *= sizeof(double);// 位移要以字节为单位！
    MPI_Status status;

    const std::string filename = pathname + "/" + file;
    if (my_pid == 0) printf("writing to %s\n", filename.c_str());

    // const idx_t diag_id = (num_diag - 1) / 2;
    assert(dof == 3);
    for (idx_t j = 0; j < ly; j++)
    for (idx_t i = 0; i < lx; i++)
    for (idx_t k = 0; k < lz; k++) {
        for (idx_t d = 0; d < num_diag; d++) {
            double * dst_ptr = buf + d * mat.elem_size + nnz_len * (k + lz * (i + lx * j));
            const data_t * src_ptr = mat.data + d * mat.elem_size + (k + mat.halo_z) * mat.slice_ed_size 
                            + (i + mat.halo_x) * mat.slice_edk_size + (j + mat.halo_y) * mat.slice_edki_size;
            for (idx_t e = 0; e < mat.elem_size; e++)
                dst_ptr[e] = src_ptr[e];
        }
        /*
        double * ptr = buf + nnz_len * (k + lz * (i + lx * j));
        double * tmp[nnz_len];
        for (idx_t p = 0; p < nnz_len; p++)
            tmp[p] = ptr[p];
        {// 第一个变量
            // 跟别的邻居结构点的第一个变量
            ptr[0] = tmp[0];
            ptr[1] = tmp[3];
            ptr[2] = tmp[6];
            ptr[3] = tmp[9];
            tmp[4] = tmp[18];
            tmp[5] = tmp[21];
            tmp[6] = tmp[24];
            // 跟自己的结构点的第二和第三个变量
            ptr[7] = tmp[10];
            ptr[8] = tmp[11];
        }
        {// 第二个变量
            // 跟别的邻居结构点的第二个变量
            ptr[9] = tmp[1];
            ptr[10]= tmp[4];
            ptr[11]= tmp[7];
            ptr[12]= tmp[13];
            ptr[13]= tmp[19];
            ptr[14]= tmp[22];
            ptr[15]= tmp[25];
            // 跟自己的邻居结构点的第一和第三个变量
            ptr[16]= tmp[12];
            ptr[17]= tmp[14];
        }
        {// 第三个变量
            // 跟别的邻居结构点的第三个变量
            ptr[18]= tmp[2];
            ptr[19]= tmp[5];
            ptr[20]= tmp[8];
            ptr[21]= tmp[17];
            tmp[22]= tmp[20];
            tmp[23]= tmp[23];
            tmp[24]= tmp[26];
            // 跟自己的邻居结构点的第一和第二个变量
            tmp[25]= tmp[15];
            tmp[26]= tmp[16];
        }
        */
    }

    // FILE* fp = fopen("check.txt", "w+");
    // for (idx_t i = 0; i < ly*lx*lz; i++) {
    //     for (idx_t j = 0; j < nnz_len; j++)
    //         fprintf(fp, "%.5e ", buf[i * nnz_len + j]);
    //     fprintf(fp, "\n");
    // }
    // fclose(fp);
    
    MPI_File_open(comm_pkg->cart_comm, filename.c_str(), 
        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, displacement, etype, write_type, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, buf, tot_len, etype, &status);
    
    MPI_File_close(&fh);
    MPI_Type_free(&write_type);

    delete buf;
}

#endif