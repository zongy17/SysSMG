#ifndef SOLID_GMG_RESTRICT_HPP
#define SOLID_GMG_RESTRICT_HPP

#include "GMG_types.hpp"

template<typename idx_t, typename data_t, int dof=NUM_DOF>
class Restrictor {
protected:
    data_t a0, a1, a2, a3;
    const RESTRICT_TYPE type;
public:
    Restrictor(RESTRICT_TYPE type): type(type)  { 
        setup_weights();
    }
    virtual void setup_weights();
    virtual void apply(const par_structVector<idx_t, data_t, dof> & par_fine_vec, 
        par_structVector<idx_t, data_t, dof> & par_coar_vec, const COAR_TO_FINE_INFO<idx_t> & info);
    virtual ~Restrictor() {  }
};

template<typename idx_t, typename data_t, int dof>
void Restrictor<idx_t, data_t, dof>::setup_weights() {
    switch (type)
    {
    case Rst_4cell:
        a0 = 0.25;
        break;
    case Rst_8cell:
        //    a00----a00
        //    /|     /|
        //  a00----a00|
        //   | a00--|a00
        //   |/     |/
        //  a00----a00
        a0 = 0.125;
        break;
    case Rst_64cell:
        /*      a3----a2----a2----a3
                /     /    /     /
              a2----a1----a1----a2
              /     /     /     /
            a2----a1----a1----a2
            /     /     /     /
          a3----a2----a2----a3
                 
                a2----a1----a1----a2
                /     /    /     /
              a1----a0----a0----a1
              /     /     /     /
            a1----a0----a0----a1
            /     /     /     /
          a2----a1----a1----a2

                 a2----a1----a1----a2
                /     /    /     /
              a1----a0----a0----a1
              /     /     /     /
            a1----a0----a0----a1
            /     /     /     /
          a2----a1----a1----a2

                 a3----a2----a2----a3
                /     /    /     /
              a2----a1----a1----a2
              /     /     /     /
            a2----a1----a1----a2
            /     /     /     /
          a3----a2----a2----a3
        */
        a0 = 0.052734375;
        a1 = 0.017578125;
        a2 = 0.005859375;
        a3 = 0.001953125;
        break;
    default:
        printf("Invalid restrictor type!\n");
        MPI_Abort(MPI_COMM_WORLD, -20221105);
    }
}

template<typename idx_t, typename data_t, int dof>
void Restrictor<idx_t, data_t, dof>::apply(const par_structVector<idx_t, data_t, dof> & par_fine_vec, 
        par_structVector<idx_t, data_t, dof> & par_coar_vec, const COAR_TO_FINE_INFO<idx_t> & info) 
{
    const seq_structVector<idx_t, data_t, dof> & f_vec = *(par_fine_vec.local_vector);
          seq_structVector<idx_t, data_t, dof> & c_vec = *(par_coar_vec.local_vector);
    CHECK_HALO(f_vec, c_vec);
    const idx_t hx = f_vec.halo_x         , hy = f_vec.halo_y         , hz = f_vec.halo_z         ;
    const idx_t bx = info.fine_base_idx[0], by = info.fine_base_idx[1], bz = info.fine_base_idx[2];
    const idx_t sx = info.stride[0]       , sy = info.stride[1]       , sz = info.stride[2]       ;

    const idx_t cibeg = c_vec.halo_x, ciend = cibeg + c_vec.local_x,
                cjbeg = c_vec.halo_y, cjend = cjbeg + c_vec.local_y,
                ckbeg = c_vec.halo_z, ckend = ckbeg + c_vec.local_z;
    const idx_t c_dk_size = c_vec.slice_dk_size, c_dki_size = c_vec.slice_dki_size;
    const idx_t f_dk_size = f_vec.slice_dk_size, f_dki_size = f_vec.slice_dki_size;
    const data_t * f_data = f_vec.data;
    data_t * c_data = c_vec.data;

#define C_VECIDX(k, i, j) (k) * dof + (i) * c_dk_size + (j) * c_dki_size
#define F_VECIDX(k, i, j) (k) * dof + (i) * f_dk_size + (j) * f_dki_size

    if (type == Rst_8cell) {
        if (bx < 0 || by < 0 || bz < 0)// 此时需要引用到不在自己进程负责范围内的数据
            par_fine_vec.update_halo();

        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t cj = cjbeg; cj < cjend; cj++)
        for (idx_t ci = cibeg; ci < ciend; ci++)
        for (idx_t ck = ckbeg; ck < ckend; ck++) {
            const idx_t fj = hy + by + (cj - cjbeg) * sy;// fj - fjbeg == (cj - cjbeg) * sy + by，而其中fjbeg就等于hy
            const idx_t fi = hx + bx + (ci - cibeg) * sx;
            const idx_t fk = hz + bz + (ck - ckbeg) * sz;
            data_t * dst = c_data + C_VECIDX(ck, ci, cj);
            const data_t* src0 = f_data + F_VECIDX(fk  , fi  , fj  ),
                        * src1 = f_data + F_VECIDX(fk+1, fi  , fj  ),
                        * src2 = f_data + F_VECIDX(fk  , fi+1, fj  ),
                        * src3 = f_data + F_VECIDX(fk+1, fi+1, fj  ),
                        * src4 = f_data + F_VECIDX(fk  , fi  , fj+1),
                        * src5 = f_data + F_VECIDX(fk+1, fi  , fj+1),
                        * src6 = f_data + F_VECIDX(fk  , fi+1, fj+1),
                        * src7 = f_data + F_VECIDX(fk+1, fi+1, fj+1);
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                dst[f] = a0 * ( src0[f] + src1[f] + src2[f] + src3[f]
                            +   src4[f] + src5[f] + src6[f] + src7[f] );
        }
    }
    else if (type == Rst_4cell) {
        if (bx < 0 || by < 0)
            par_fine_vec.update_halo();
        
        assert(c_dk_size == f_dk_size);
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t cj = cjbeg; cj < cjend; cj++)
        for (idx_t ci = cibeg; ci < ciend; ci++)
        for (idx_t ck = ckbeg; ck < ckend; ck++) {// same as fk's range
            const idx_t fj = hy + by + (cj - cjbeg) * sy;// fj - fjbeg == (cj - cjbeg) * sy + by，而其中fjbeg就等于hy
            const idx_t fi = hx + bx + (ci - cibeg) * sx;
            const idx_t fk = ck;
            data_t * dst = c_data + C_VECIDX(ck, ci, cj);
            const data_t* src0 = f_data + F_VECIDX(fk  , fi  , fj  ),
                        * src1 = f_data + F_VECIDX(fk  , fi+1, fj  ),
                        * src2 = f_data + F_VECIDX(fk  , fi  , fj+1),
                        * src3 = f_data + F_VECIDX(fk  , fi+1, fj+1);
            #pragma GCC unroll (4)
            for (idx_t f = 0; f < dof; f++)
                dst[f] = a0 * ( src0[f] + src1[f] + src2[f] + src3[f] );
        }
    }
    else if (type == Rst_64cell) {
        par_fine_vec.update_halo();

        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t cj = cjbeg; cj < cjend; cj++)
        for (idx_t ci = cibeg; ci < ciend; ci++)
        for (idx_t ck = ckbeg; ck < ckend; ck++) {
            const idx_t fj = hy + by + (cj - cjbeg) * sy;// fj - fjbeg == (cj - cjbeg) * sy + by，而其中fjbeg就等于hy
            const idx_t fi = hx + bx + (ci - cibeg) * sx;
            const idx_t fk = hz + bz + (ck - ckbeg) * sz;

            data_t * dst = c_data + C_VECIDX(ck, ci, cj);
            const data_t* src0 = f_data + F_VECIDX(fk  , fi  , fj  ),
                        * src1 = f_data + F_VECIDX(fk+1, fi  , fj  ),
                        * src2 = f_data + F_VECIDX(fk  , fi+1, fj  ),
                        * src3 = f_data + F_VECIDX(fk+1, fi+1, fj  ),
                        * src4 = f_data + F_VECIDX(fk  , fi  , fj+1),
                        * src5 = f_data + F_VECIDX(fk+1, fi  , fj+1),
                        * src6 = f_data + F_VECIDX(fk  , fi+1, fj+1),
                        * src7 = f_data + F_VECIDX(fk+1, fi+1, fj+1),

                *src8  = f_data + F_VECIDX(fk-1, fi  , fj  ), *src9  = f_data + F_VECIDX(fk+2, fi  , fj  ),
                *src10 = f_data + F_VECIDX(fk-1, fi+1, fj  ), *src11 = f_data + F_VECIDX(fk+2, fi+1, fj  ),
                *src12 = f_data + F_VECIDX(fk-1, fi  , fj+1), *src13 = f_data + F_VECIDX(fk+2, fi  , fj+1),
                *src14 = f_data + F_VECIDX(fk-1, fi+1, fj+1), *src15 = f_data + F_VECIDX(fk+2, fi+1, fj+1),

                *src16 = f_data + F_VECIDX(fk  , fi-1, fj  ), *src17 = f_data + F_VECIDX(fk  , fi+2, fj  ),
                *src18 = f_data + F_VECIDX(fk  , fi  , fj-1), *src19 = f_data + F_VECIDX(fk  , fi  , fj+2),
                *src20 = f_data + F_VECIDX(fk  , fi+2, fj+1), *src21 = f_data + F_VECIDX(fk  , fi+1, fj+2),
                *src22 = f_data + F_VECIDX(fk  , fi-1, fj+1), *src23 = f_data + F_VECIDX(fk  , fi+1, fj-1),

                *src24 = f_data + F_VECIDX(fk+1, fi-1, fj  ), *src25 = f_data + F_VECIDX(fk+1, fi+2, fj  ),
                *src26 = f_data + F_VECIDX(fk+1, fi  , fj-1), *src27 = f_data + F_VECIDX(fk+1, fi  , fj+2),
                *src28 = f_data + F_VECIDX(fk+1, fi+2, fj+1), *src29 = f_data + F_VECIDX(fk+1, fi+1, fj+2),
                *src30 = f_data + F_VECIDX(fk+1, fi-1, fj+1), *src31 = f_data + F_VECIDX(fk+1, fi+1, fj-1),

                *src32 = f_data + F_VECIDX(fk  , fi-1, fj-1), *src33 = f_data + F_VECIDX(fk+1, fi-1, fj-1),
                *src34 = f_data + F_VECIDX(fk  , fi+2, fj-1), *src35 = f_data + F_VECIDX(fk+1, fi+2, fj-1),
                *src36 = f_data + F_VECIDX(fk  , fi-1, fj+2), *src37 = f_data + F_VECIDX(fk+1, fi-1, fj+2),
                *src38 = f_data + F_VECIDX(fk  , fi+2, fj+2), *src39 = f_data + F_VECIDX(fk+1, fi+2, fj+2),

                *src40 = f_data + F_VECIDX(fk-1, fi-1, fj  ), *src41 = f_data + F_VECIDX(fk-1, fi  , fj-1),
                *src42 = f_data + F_VECIDX(fk-1, fi+2, fj  ), *src43 = f_data + F_VECIDX(fk-1, fi+1, fj-1),
                *src44 = f_data + F_VECIDX(fk-1, fi-1, fj+1), *src45 = f_data + F_VECIDX(fk-1, fi  , fj+2),
                *src46 = f_data + F_VECIDX(fk-1, fi+2, fj+1), *src47 = f_data + F_VECIDX(fk-1, fi+1, fj+2),

                *src48 = f_data + F_VECIDX(fk+2, fi-1, fj  ), *src49 = f_data + F_VECIDX(fk+2, fi  , fj-1),
                *src50 = f_data + F_VECIDX(fk+2, fi+2, fj  ), *src51 = f_data + F_VECIDX(fk+2, fi+1, fj-1),
                *src52 = f_data + F_VECIDX(fk+2, fi-1, fj+1), *src53 = f_data + F_VECIDX(fk+2, fi  , fj+2),
                *src54 = f_data + F_VECIDX(fk+2, fi+2, fj+1), *src55 = f_data + F_VECIDX(fk+2, fi+1, fj+2),

                *src56 = f_data + F_VECIDX(fk-1, fi-1, fj-1), *src57 = f_data + F_VECIDX(fk+2, fi-1, fj-1),
                *src58 = f_data + F_VECIDX(fk-1, fi+2, fj-1), *src59 = f_data + F_VECIDX(fk+2, fi+2, fj-1),
                *src60 = f_data + F_VECIDX(fk-1, fi-1, fj+2), *src61 = f_data + F_VECIDX(fk+2, fi-1, fj+2),
                *src62 = f_data + F_VECIDX(fk-1, fi+2, fj+2), *src63 = f_data + F_VECIDX(fk+2, fi+2, fj+2);
            
            for (idx_t f = 0; f < dof; f++)
                dst[f] = a0 * (src0 [f] + src1[f] + src2 [f] + src3 [f] + src4 [f] + src5 [f] + src6 [f] + src7 [f])
                    +    a1 * (src8 [f] + src9[f] + src10[f] + src11[f] + src12[f] + src13[f] + src14[f] + src15[f]
                            +  src16[f]+ src17[f] + src18[f] + src19[f] + src20[f] + src21[f] + src22[f] + src23[f]
                            +  src24[f]+ src25[f] + src26[f] + src27[f] + src28[f] + src29[f] + src30[f] + src31[f] )
                    +    a2 * (src32[f]+ src33[f] + src34[f] + src35[f] + src36[f] + src37[f] + src38[f] + src39[f]
                            +  src40[f]+ src41[f] + src42[f] + src43[f] + src44[f] + src45[f] + src46[f] + src47[f]
                            +  src48[f]+ src49[f] + src50[f] + src51[f] + src52[f] + src53[f] + src54[f] + src55[f] )
                    +    a3 * (src56[f]+ src57[f] + src58[f] + src59[f] + src60[f] + src61[f] + src62[f] + src63[f] );
        }
    }
    else {
        assert(false);
    }
#undef C_VECIDX
#undef F_VECIDX
}

#endif