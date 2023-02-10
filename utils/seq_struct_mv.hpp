#ifndef SOLID_SEQ_STRUCT_MV_HPP
#define SOLID_SEQ_STRUCT_MV_HPP

#include "common.hpp"

// 向量只需要一种精度
template<typename idx_t, typename data_t, int dof=NUM_DOF>
class seq_structVector {
public:
    const idx_t local_x;// lon方向的格点数(仅含计算区域)
    const idx_t local_y;// lat方向的格点数(仅含计算区域)
    const idx_t local_z;// 垂直方向的格点数(仅含计算区域)
    const idx_t halo_x;// lon方向的halo区宽度
    const idx_t halo_y;// lat方向的halo区宽度
    const idx_t halo_z;// 垂直方向的halo区宽度
    data_t * data;

    // 数据存储顺序从内到外为(dof, k, j, i)
    idx_t slice_dk_size;
    idx_t slice_dki_size;

    seq_structVector(idx_t lx, idx_t ly, idx_t lz, idx_t hx, idx_t hy, idx_t hz);
    // 拷贝构造函数，开辟同样规格的data
    seq_structVector(const seq_structVector & model);
    ~seq_structVector();

    void init_debug(idx_t off_x, idx_t off_y, idx_t off_z);
    void print_level(idx_t ilev);

    void operator=(data_t val);
    void set_halo(data_t val);
};


// 矩阵需要两种精度：数据的存储精度data_t，和计算时的精度calc_t
template<typename idx_t, typename data_t, typename calc_t, int dof=NUM_DOF>
class seq_structMatrix {
public:
    idx_t num_diag;// 矩阵对角线数（19）
    idx_t local_x;// lon方向的格点数(仅含计算区域)
    idx_t local_y;// lat方向的格点数(仅含计算区域)
    idx_t local_z;// 垂直方向的格点数(仅含计算区域)
    idx_t halo_x;// lon方向的halo区宽度
    idx_t halo_y;// lat方向的halo区宽度
    idx_t halo_z;// 垂直方向的halo区宽度
    data_t * data;

    // 数据存储顺序从内到外为(dof, diag, k, j, i)
    const idx_t elem_size = dof * dof;// local matrix (as an element) size
    idx_t slice_ed_size;
    idx_t slice_edk_size;
    idx_t slice_edki_size;

    seq_structMatrix(idx_t num_d, idx_t lx, idx_t ly, idx_t lz, idx_t hx, idx_t hy, idx_t hz);
    // 拷贝构造函数，开辟同样规格的data
    seq_structMatrix(const seq_structMatrix & model);
    ~seq_structMatrix();

    void init_debug(idx_t off_x, idx_t off_y, idx_t off_z);
    void print_level_diag(idx_t ilev, idx_t idiag);
    void operator=(data_t val);
    void set_diag_val(idx_t d, data_t val);

    void extract_diag(idx_t idx_diag) const;
    void truncate() {
#ifdef __aarch64__
        idx_t tot_len = (2 * halo_y + local_y) * (2 * halo_x + local_x) * (2 * halo_z + local_z) * num_diag * elem_size;
        for (idx_t i = 0; i < tot_len; i++) {
            __fp16 tmp = (__fp16) data[i];
            // if (i == local_x * local_y * local_z * 4) printf("seqMat truncate %.20e to", data[i]);
            data[i] = (data_t) tmp;
            // if (i == local_x * local_y * local_z * 4) printf("%.20e\n", data[i]);
        }
#else
        printf("architecture not support truncated to fp16\n");
#endif
    }

    // 矩阵接受的SpMV运算是以操作精度的
    void Mult(const seq_structVector<idx_t, calc_t, dof> & x, seq_structVector<idx_t, calc_t, dof> & y,
            const seq_structVector<idx_t, calc_t, dof> * sqrtD_ptr = nullptr) const;
    void (*spmv)(const idx_t num, const idx_t vec_dk_size, const idx_t vec_dki_size,
        const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const calc_t * sqD_jik) = nullptr;
    void (*spmv_scaled)(const idx_t num, const idx_t vec_dk_size, const idx_t vec_dki_size,
        const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const calc_t * sqD_jik) = nullptr;
};


/*
 * * * * * seq_structVetor * * * * * 
 */

template<typename idx_t, typename data_t, int dof>
seq_structVector<idx_t, data_t, dof>::seq_structVector(idx_t lx, idx_t ly, idx_t lz, idx_t hx, idx_t hy, idx_t hz)
    : local_x(lx), local_y(ly), local_z(lz), halo_x(hx), halo_y(hy), halo_z(hz)
{
    idx_t   tot_x = local_x + 2 * halo_x,
            tot_y = local_y + 2 * halo_y,
            tot_z = local_z + 2 * halo_z;
    data = new data_t[tot_x * tot_y * tot_z * dof];
#ifdef DEBUG
    for (idx_t i = 0; i < tot_x * tot_y * tot_z; i++) data[i] = -99;
#endif
    slice_dk_size  = (local_z + 2 * halo_z) * dof;
    slice_dki_size = slice_dk_size * (local_x + 2 * halo_x); 
}

template<typename idx_t, typename data_t, int dof>
seq_structVector<idx_t, data_t, dof>::seq_structVector(const seq_structVector & model)
    : local_x(model.local_x), local_y(model.local_y), local_z(model.local_z),
      halo_x (model.halo_x) , halo_y (model.halo_y ), halo_z (model.halo_z) , 
      slice_dk_size(model.slice_dk_size), slice_dki_size(model.slice_dki_size)
{
    idx_t   tot_x = local_x + 2 * halo_x,
            tot_y = local_y + 2 * halo_y,
            tot_z = local_z + 2 * halo_z;
    data = new data_t[tot_x * tot_y * tot_z * dof];
}

template<typename idx_t, typename data_t, int dof>
seq_structVector<idx_t, data_t, dof>::~seq_structVector() {
    delete data;
    data = nullptr;
}

template<typename idx_t, typename data_t, int dof>
void seq_structVector<idx_t, data_t, dof>::init_debug(idx_t off_x, idx_t off_y, idx_t off_z) 
{
    idx_t tot = slice_dki_size * (local_y + 2 * halo_y);
    for (idx_t i = 0; i < tot; i++)
        data[i] = 0.0;

    idx_t xbeg = halo_x, xend = xbeg + local_x,
            ybeg = halo_y, yend = ybeg + local_y,
            zbeg = halo_z, zend = zbeg + local_z;
    for (idx_t j = ybeg; j < yend; j++)
    for (idx_t i = xbeg; i < xend; i++)
    for (idx_t k = zbeg; k < zend; k++) {
        for (idx_t f = 0; f < dof; f++)
            data[f + k * dof + i * slice_dk_size + j * slice_dki_size] 
                = 100.0 * (off_x + i - xbeg) + off_y + j - ybeg + 1e-2 * (off_z + k - zbeg);
    }
}

template<typename idx_t, typename data_t, typename res_t, int dof=NUM_DOF>
res_t seq_vec_dot(const seq_structVector<idx_t, data_t, dof> & x, const seq_structVector<idx_t, data_t, dof> & y) {
    CHECK_LOCAL_HALO(x, y);
    assert(x.slice_dk_size == y.slice_dk_size);// make sure x.dof == y.dof

    const idx_t xbeg = x.halo_x, xend = xbeg + x.local_x,
                ybeg = x.halo_y, yend = ybeg + x.local_y,
                zbeg = x.halo_z, zend = zbeg + x.local_z;
    const idx_t slice_dk_size = x.slice_dk_size, slice_dki_size = x.slice_dki_size;
    res_t dot = 0.0;

    #pragma omp parallel for collapse(2) reduction(+:dot) schedule(static)
    for (idx_t j = ybeg; j < yend; j++)
    for (idx_t i = xbeg; i < xend; i++) {
        idx_t ji_loc = j * slice_dki_size + i * slice_dk_size;
        const data_t * x_data = x.data + ji_loc, * y_data = y.data + ji_loc;
        for (idx_t k = zbeg; k < zend; k++) {
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                dot += (res_t) x_data[k * dof + f] * (res_t) y_data[k * dof + f];
        }            
    }
    return dot;
}

template<typename idx_t, typename data_t, typename scalar_t, int dof>
void seq_vec_add(const seq_structVector<idx_t, data_t, dof> & v1, scalar_t alpha, 
                 const seq_structVector<idx_t, data_t, dof> & v2, seq_structVector<idx_t, data_t, dof> & v) 
{
    CHECK_LOCAL_HALO(v1, v2);
    CHECK_LOCAL_HALO(v1, v );
    assert(v1.slice_dk_size == v2.slice_dk_size && v1.slice_dk_size == v.slice_dk_size);
    
    const idx_t ibeg = v1.halo_x, iend = ibeg + v1.local_x,
                jbeg = v1.halo_y, jend = jbeg + v1.local_y,
                kbeg = v1.halo_z, kend = kbeg + v1.local_z;
    const idx_t vec_dk_size = v1.slice_dk_size, vec_dki_size = v1.slice_dki_size;

    #pragma omp parallel for collapse(2) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        idx_t ji_loc = j * vec_dki_size + i * vec_dk_size;
        const data_t * v1_data = v1.data + ji_loc, * v2_data = v2.data + ji_loc;
        data_t * v_data = v.data + ji_loc;
        for (idx_t k = kbeg; k < kend; k++) {
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                v_data[k * dof + f] = v1_data[k * dof + f] + alpha * v2_data[k * dof + f];
        }
            
    }
}

template<typename idx_t, typename data_t, int dof>
void seq_vec_copy(const seq_structVector<idx_t, data_t, dof> & src, seq_structVector<idx_t, data_t, dof> & dst)
{
    CHECK_LOCAL_HALO(src, dst);
    assert(src.slice_dk_size == dst.slice_dk_size);
    
    const idx_t ibeg = src.halo_x, iend = ibeg + src.local_x,
                jbeg = src.halo_y, jend = jbeg + src.local_y,
                kbeg = src.halo_z, kend = kbeg + src.local_z;
    const idx_t vec_dk_size = src.slice_dk_size, vec_dki_size = src.slice_dki_size;

    #pragma omp parallel for collapse(2) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        idx_t ji_loc = j * vec_dki_size + i * vec_dk_size;
        const data_t * src_data = src.data + ji_loc;
        data_t * dst_data = dst.data + ji_loc;
        for (idx_t k = kbeg; k < kend; k++) {
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                dst_data[k * dof + f] = src_data[k * dof + f];
        }
    }
}

template<typename idx_t, typename data_t, int dof>
void seq_structVector<idx_t, data_t, dof>::operator=(data_t val) {
    idx_t   xbeg = halo_x, xend = xbeg + local_x,
            ybeg = halo_y, yend = ybeg + local_y,
            zbeg = halo_z, zend = zbeg + local_z;
    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = ybeg; j < yend; j++)
    for (idx_t i = xbeg; i < xend; i++) {
        idx_t ji_loc = j * slice_dki_size + i * slice_dk_size;
        data_t * ptr = data + ji_loc;
        for (idx_t k = zbeg; k < zend; k++) {
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                ptr[k * dof + f] = val;
        }
    }
}

template<typename idx_t, typename data_t, int dof>
void seq_structVector<idx_t, data_t, dof>::set_halo(data_t val) {
    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = 0; j < halo_y; j++)
    for (idx_t i = 0; i < halo_x * 2 + local_x; i++) {
        idx_t ji_loc = j * slice_dki_size + i * slice_dk_size;
        data_t * ptr = data + ji_loc;
        for (idx_t k = 0; k < halo_z * 2 + local_z; k++)
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                ptr[k * dof + f] = val;
    }

    for (idx_t j = halo_y; j < halo_y + local_y; j++)
    for (idx_t i = 0; i < halo_x; i++) {
        idx_t ji_loc = j * slice_dki_size + i * slice_dk_size;
        data_t * ptr = data + ji_loc;
        for (idx_t k = 0; k < halo_z * 2 + local_z; k++)
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                ptr[k * dof + f] = val;
    }
        
    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = halo_y; j < halo_y + local_y; j++) 
    for (idx_t i = halo_x; i < halo_x + local_x; i++) {
        idx_t ji_loc = j * slice_dki_size + i * slice_dk_size;
        data_t * ptr = data + ji_loc;
        for (idx_t k = 0; k < halo_z; k++)
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                ptr[k * dof + f] = val;
        for (idx_t k = halo_z + local_z; k < halo_z * 2 + local_z; k++)
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                ptr[k * dof + f] = val;
    }

    for (idx_t j = halo_y; j < halo_y + local_y; j++)
    for (idx_t i = halo_x + local_x; i < halo_x * 2 + local_x; i++) {
        idx_t ji_loc = j * slice_dki_size + i * slice_dk_size;
        data_t * ptr = data + ji_loc;
        for (idx_t k = 0; k < halo_z * 2 + local_z; k++)
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                ptr[k * dof + f] = val;
    }

    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = halo_y + local_y; j < halo_y * 2 + local_y; j++)
    for (idx_t i = 0; i < halo_x * 2 + local_x; i++) {
        idx_t ji_loc = j * slice_dki_size + i * slice_dk_size;
        data_t * ptr = data + ji_loc;
        for (idx_t k = 0; k < halo_z * 2 + local_z; k++)
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                ptr[k * dof + f] = val;
    }
}

template<typename idx_t, typename data_t, typename scalar_t, int dof>
void seq_vec_mul_by_scalar(const scalar_t coeff, const seq_structVector<idx_t, data_t, dof> & src, 
    seq_structVector<idx_t, data_t, dof> & dst) 
{
    CHECK_LOCAL_HALO(src, dst);
    assert(dst.slice_dk_size == src.slice_dk_size);
    
    const idx_t ibeg = src.halo_x, iend = ibeg + src.local_x,
                jbeg = src.halo_y, jend = jbeg + src.local_y,
                kbeg = src.halo_z, kend = kbeg + src.local_z;
    const idx_t vec_dk_size = src.slice_dk_size, vec_dki_size = src.slice_dki_size;

    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        idx_t ji_loc = j * vec_dki_size + i * vec_dk_size;
        data_t * dst_data = dst.data + ji_loc;
        const data_t * src_data = src.data + ji_loc;
        for (idx_t k = kbeg; k < kend; k++)
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                dst_data[k * dof + f] = coeff * src_data[k * dof + f];
    }
}

template<typename idx_t, typename data_t, typename scalar_t, int dof>
void seq_vec_scale(const scalar_t coeff, seq_structVector<idx_t, data_t, dof> & vec) {
    data_t * data = vec.data;
    const idx_t ibeg = vec.halo_x, iend = ibeg + vec.local_x,
                jbeg = vec.halo_y, jend = jbeg + vec.local_y,
                kbeg = vec.halo_z, kend = kbeg + vec.local_z;
    const idx_t vec_dk_size = vec.slice_dk_size, vec_dki_size = vec.slice_dki_size;

    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        idx_t ji_loc = j * vec_dki_size + i * vec_dk_size;
        data_t * ptr = data + ji_loc;
        for (idx_t k = kbeg; k < kend; k++)
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                ptr[k * dof + f] *= coeff;
    }
}

template<typename idx_t, typename data_t1, typename data_t2, int dof>
void seq_vec_elemwise_div(seq_structVector<idx_t, data_t1, dof> & inout_vec, const seq_structVector<idx_t, data_t2, dof> & scaleplate)
{
    CHECK_LOCAL_HALO(inout_vec, scaleplate);
    const idx_t jbeg = inout_vec.halo_y, jend = jbeg + inout_vec.local_y,
                ibeg = inout_vec.halo_x, iend = ibeg + inout_vec.local_x,
                kbeg = inout_vec.halo_z, kend = kbeg + inout_vec.local_z;
    const idx_t vec_dki_size = inout_vec.slice_dki_size, vec_dk_size = inout_vec.slice_dk_size;
    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) 
    for (idx_t k = kbeg; k < kend; k++) {
        idx_t offset = j * vec_dki_size + i * vec_dk_size + k * dof;
        data_t1 * dst_ptr = inout_vec.data + offset;
        const data_t2 * src_ptr = scaleplate.data + offset;
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            dst_ptr[f] /= src_ptr[f];
    }
}

/*
 * * * * * seq_structMatrix * * * * * 
 */
#include "scal_kernels.hpp"
#include "vect_kernels.hpp"

template<typename idx_t, typename data_t, typename calc_t, int dof>
seq_structMatrix<idx_t, data_t, calc_t, dof>::seq_structMatrix(idx_t num_d, idx_t lx, idx_t ly, idx_t lz, idx_t hx, idx_t hy, idx_t hz)
    : num_diag(num_d), local_x(lx), local_y(ly), local_z(lz), halo_x(hx), halo_y(hy), halo_z(hz)
{
    idx_t tot = elem_size * num_diag * (local_x + 2 * halo_x) * (local_y + 2 * halo_y) * (local_z + 2 * halo_z);
    data = new data_t[tot];
#ifdef DEBUG
    for (idx_t i = 0; i < tot; i++) data[i] = -9999.9;
#endif
    slice_ed_size = elem_size * num_diag;
    slice_edk_size  = slice_ed_size * (local_z + 2 * halo_z);
    slice_edki_size = slice_edk_size * (local_x + 2 * halo_x);
    switch (num_diag)
    {
    case  7:
        if constexpr (sizeof(data_t) == 2 && sizeof(calc_t) == 4) {
            spmv        = AOS_spmv_3d_Cal32Stg16<dof, 7>;
        } else {
            spmv        = AOS_spmv_3d_normal<idx_t, data_t, calc_t, dof, 7>;
        }
        break;
    default: break;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
seq_structMatrix<idx_t, data_t, calc_t, dof>::seq_structMatrix(const seq_structMatrix & model)
    : num_diag(model.num_diag),
      local_x(model.local_x), local_y(model.local_y), local_z(model.local_z),
      halo_x (model.halo_x) , halo_y (model.halo_y ), halo_z (model.halo_z) , 
      slice_ed_size(model.slice_ed_size), slice_edk_size(model.slice_edk_size), slice_edki_size(model.slice_edki_size)
{
    idx_t tot = elem_size * num_diag * (local_x + 2 * halo_x) * (local_y + 2 * halo_y) * (local_z + 2 * halo_z);
    data = new data_t[tot];
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
seq_structMatrix<idx_t, data_t, calc_t, dof>::~seq_structMatrix() {
    delete data;
    data = nullptr;
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void seq_structMatrix<idx_t, data_t, calc_t, dof>::Mult(const seq_structVector<idx_t, calc_t, dof> & x,
        seq_structVector<idx_t, calc_t, dof> & y, const seq_structVector<idx_t, calc_t, dof> * sqrtD_ptr) const
{
    CHECK_LOCAL_HALO(*this, x);
    CHECK_LOCAL_HALO(x , y);
    const data_t* mat_data = data;
    const calc_t* aux_data = (sqrtD_ptr) ? sqrtD_ptr->data : nullptr;
    void (*kernel)(const idx_t num, const idx_t vec_dk_size, const idx_t vec_dki_size,
        const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const calc_t * sqD_jik) = sqrtD_ptr ? spmv_scaled : spmv;
    assert(kernel);

    const calc_t * x_data = x.data;
    calc_t * y_data = y.data;

    idx_t   ibeg = halo_x, iend = ibeg + local_x,
            jbeg = halo_y, jend = jbeg + local_y,
            kbeg = halo_z, kend = kbeg + local_z;
    idx_t vec_dk_size = x.slice_dk_size, vec_dki_size = x.slice_dki_size;
    const idx_t col_height = kend - kbeg;

    #pragma omp parallel for collapse(2) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        const data_t * A_jik = mat_data + j * slice_edki_size + i * slice_edk_size + kbeg * slice_ed_size;
        const idx_t vec_off = j * vec_dki_size + i * vec_dk_size + kbeg * dof;
        const calc_t * aux_jik = (sqrtD_ptr) ? (aux_data + vec_off) : nullptr;
        const calc_t * x_jik = x_data + vec_off;
        calc_t * y_jik = y_data + vec_off;
        kernel(col_height, vec_dk_size, vec_dki_size, A_jik, x_jik, y_jik, aux_jik);
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void seq_structMatrix<idx_t, data_t, calc_t, dof>::operator=(data_t val) 
{
    for (idx_t j = 0; j < halo_y * 2 + local_y; j++) 
    for (idx_t i = 0; i < halo_x * 2 + local_x; i++)
    for (idx_t k = 0; k < halo_z * 2 + local_z; k++) 
    for (idx_t d = 0; d < num_diag; d++) {
        data_t * ptr = data + d * elem_size + k * slice_ed_size + i * slice_edk_size + j * slice_edki_size;
        #pragma GCC unroll (4)
        for (idx_t e = 0; e < elem_size; e++)
            ptr[e] = val;
    }
}

template<typename idx_t, typename data_t, typename calc_t, int dof>
void seq_structMatrix<idx_t, data_t, calc_t, dof>::set_diag_val(idx_t d, data_t val) 
{
    const idx_t jbeg = halo_y, jend = jbeg + local_y,
                ibeg = halo_x, iend = ibeg + local_x,
                kbeg = halo_z, kend = kbeg + local_z;
    for (idx_t j = jbeg; j < jend; j++) 
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        data_t * ptr = data + d * elem_size + k * slice_ed_size + i * slice_edk_size + j * slice_edki_size;
        #pragma GCC unroll (4)
        for (idx_t e = 0; e < elem_size; e++)
            ptr[e] = 0.0;
        #pragma GCC unroll (4)
        for (idx_t f = 0; f < dof; f++)
            ptr[f * dof + f] = val;
    }
}

#endif