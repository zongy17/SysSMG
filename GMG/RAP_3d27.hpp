#ifndef SOLID_RAP_3D27_HPP
#define SOLID_RAP_3D27_HPP

#include "../utils/common.hpp"

/* assume 粗网格也是3d27，所以遍历27个非零元
                  /20--- 23------26
                 11|    14      17|
               2---|---5-------8  |
               |   |           |  |
               |   19----22----|-25
   z   y       | 10|    13     |16|
   ^  ^        1---|-- 4 ------7  |
   | /         |   |           |  |
   |/          |   18----21----|-24 
   O-------> x | 9      12     |15 
               0/------3-------6/

    
    这样排布的好处
    外维j，中间i，内维k，三维坐标(i,j,k) 的一维寻址为 i*3 + j*9 + k + 13(数组整体偏移)，其中13是由27减1再除以2得到的
    0 <=> (i,j,k)=(-1,-1,-1) 取相反数 ( 1, 1, 1) <=> 26
    1 <=> (i,j,k)=(-1,-1, 0) 取相反数 ( 1, 1, 0) <=> 25
    2 <=> (i,j,k)=(-1,-1, 1) 取相反数 ( 1, 1,-1) <=> 24
    3 <=> (i,j,k)=( 0,-1,-1) 取相反数 ( 0, 1, 1) <=> 23
    4 <=> (i,j,k)=( 0,-1, 0) 取相反数 ( 0, 1, 0) <=> 22
    10<=> (i,j,k)=(-1, 0, 0) 取相反数 ( 1, 0, 0) <=> 16
    12<=> (i,j,k)=( 0, 0,-1) 取相反数 ( 0, 0, 1) <=> 14
    这也就意味着当知道邻居相对于我的偏移（即位于0~26的哪个位置）时，可以直接取一组相反数，就得到我在邻居行的对应列的元素
 */

/*
  README:
  输入前数组fine_mat必须先做update_halo()，该函数忽略通信相关，只视作单进程内粗化。
  fx, fy, fz为fine_mat的总大小（包含halo区宽度），cx, cy, cz为coar_mat的总大小（包含halo区宽度），
  需要全局大小是因为做数组偏移寻址。
  [cibeg, ciend), [ckbeg, ckend) 是本进程粗网格的起止范围（对应于本函数中需要进行处理的粗矩阵coar_mat
  的点范围）
  base_x, base_z是本进程粗网格的第一个（左下角）点（下图中Ⓒ），对应于本进程细网格的第一个（左下角）点
  在总的细网格（包含halo区宽度）中的偏移，注意区别于外层数据结构COAR_TO_FINE_INFO中的fine_base_idx[3]
  如下图示例中：
    cibeg  = 1 (= AC.halo_x), ciend = 7 (= cibeg + AC.local_x), 
    base_x = 1 (= fine_base_idx[0] + AF.halo_x = 0 + 1), 
     
    |< - - - - - - - - - - - - - - - cx  - - - - - - - - - - - - - - - - >|
       |<-------------------------  fx  ------------------------------->|
    C  |    | C  |    | C  |    | C  |    | C  |    | C  |    | C  |    | C      肆
     --F----F----F----F----F----F----F----F----F----F----F----F----F----F--  7 
       |  ==|====|====|====|====|====|====|====|====|====|====|====|==  |    
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  6
    C  | || | C  |    | C  |    | C  |    | C  |    | C  |    | C  | || | C      叁
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  5
       | || |    |    |    |    |    |    |    |    |    |    |    | || |
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  4
    C  | || | C  |    | C  |    | C  |    | C  |    | C  |    | C  | || | C      贰
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  3
       | || |    |    |    |    |    |    |    |    |    |    |    | || |
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  2
    C  | || | C  |    | C  |    | C  |    | C  |    | C  |    | C  | || | C      壹
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  1
       |  ==|====|====|====|====|====|====|====|====|====|====|====|==  |
     --F----F----F----F----F----F----F----F----F----F----F----F----F----F--  0 
    C  |    | C  |    | C  |    | C  |    | C  |    | C  |    | C  |    | C      零
       0    1    2    3    4    5    6    7    8    9    10   11   12   13
    零        壹        贰         叁        肆        伍        陆        柒
              ^                                                 ^
              |                                                 |
            cibeg                                            ciend-1

    粗网格点用(I,J,K)表示，细网格点用(i,j,k)表示
    在不计halo区时，粗细映射关系为，I的最邻近左右为2*I, 2*I+1，J的最邻近前后为2*J, 2*J+1，K的最邻近上下为2*K, 2*K+1
    在粗细网格均有宽度为bx, by, bz的halo区时，需要加上对应的偏移
    I的最邻近左右为2*(I-bx)+bx, 2*(I-bx)+bx+1，即2*I-bx, 2*I-bx+1，同理，
    J的最邻近前后为2*(J-by)+by, 2*(J-by)+by+1，即2*J-by, 2*J-by+1
    K的最邻近上下为2*(K-bz)+bz, 2*(K-bz)+bz+1，即2*K-bz, 2*K-bz+1
 */
#define NUM_DIAG 27

template<typename idx_t, typename data_t, int dof>
void RAP_3d27(
    // 细网格的数据，以及各维度的存储大小（含halo区）
    const data_t * fine_mat, const idx_t fx, const idx_t fy, const idx_t fz, 
    // 粗网格的数据，以及各维度的存储大小（含halo区）
        data_t * coar_mat, const idx_t cx, const idx_t cy, const idx_t cz,
    // 本进程在粗网格上实际负责的范围：[cibeg, ciend) x [cjbeg, cjend) x [ckbeg, ckend)
    const idx_t cibeg, const idx_t ciend, const idx_t cjbeg, const idx_t cjend, const idx_t ckbeg, const idx_t ckend,
    // ilb, iub 分别记录本进程在i维度是否在左边界和右边界
    const bool ilb , const bool iub , 
    // jlb, jub 分别记录本进程在j维度是否在前边界和后边界
    const bool jlb , const bool jub , 
    // klb, kub 分别记录本进程在k维度是否在下边界和上边界
    const bool klb , const bool kub ,
    // 实际就是halo区的宽度，要求粗、细网格的宽度相同
    const idx_t base_x, const idx_t base_y, const idx_t base_z) 
{
	const int DOF2 = dof * dof;
    assert(base_x == 1 && base_y == 1 && base_z == 1);
    for (idx_t i = 0; i < cx * cy * cz * NUM_DIAG * DOF2; i++) {
        // 初值赋为0，为了边界条件
        coar_mat[i] = 0.0;
    }
	
    const data_t (*AF)[NUM_DIAG *DOF2] = (data_t (*)[NUM_DIAG *DOF2])fine_mat;
    data_t (*AC)[NUM_DIAG *DOF2] = (data_t (*)[NUM_DIAG *DOF2])coar_mat;

    const idx_t czcx = cz * cx;// cz * cx
    const idx_t fzfx = fz * fx;// fz * fx
#define C_IDX(i, j, k)   (k) + (i) * cz + (j) * czcx
#define F_IDX(i, j, k)   (k) + (i) * fz + (j) * fzfx
#define SHIFT(oi, oj, ok) (13 + (ok) + (oi) * 3 + (oj) * 9) * DOF2 // 最内维为自由度维
// 当本进程不在某维度的边界时，可以直接算（前提要求细网格的halo区已填充，但R采用4点时似乎也不用？）
#define CHECK_BDR(CI, CJ, CK) \
    (ilb==false || (CI) >= cibeg) && (iub==false || (CI) < ciend) && \
    (jlb==false || (CJ) >= cjbeg) && (jub==false || (CJ) < cjend) && \
    (klb==false || (CK) >= ckbeg) && (kub==false || (CK) < ckend)

	#pragma omp parallel for collapse(3) schedule(static)
    for (idx_t J = cjbeg; J < cjend; J++)
    for (idx_t I = cibeg; I < ciend; I++)
    for (idx_t K = ckbeg; K < ckend; K++) {

		if (CHECK_BDR(I - 1,J - 1,K - 1)) {// ingb=0
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0156250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 0 * DOF2);
		}
		if (CHECK_BDR(I - 1,J - 1,K)) {// ingb=1
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 1 * DOF2);
		}
		if (CHECK_BDR(I - 1,J - 1,K + 1)) {// ingb=2
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 2 * DOF2);
		}
		if (CHECK_BDR(I,J - 1,K - 1)) {// ingb=3
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 3 * DOF2);
		}
		if (CHECK_BDR(I,J - 1,K)) {// ingb=4
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 4 * DOF2);
		}
		if (CHECK_BDR(I,J - 1,K + 1)) {// ingb=5
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 5 * DOF2);
		}
		if (CHECK_BDR(I + 1,J - 1,K - 1)) {// ingb=6
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0156250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 6 * DOF2);
		}
		if (CHECK_BDR(I + 1,J - 1,K)) {// ingb=7
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 7 * DOF2);
		}
		if (CHECK_BDR(I + 1,J - 1,K + 1)) {// ingb=8
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 8 * DOF2);
		}
		if (CHECK_BDR(I - 1,J,K - 1)) {// ingb=9
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 9 * DOF2);
		}
		if (CHECK_BDR(I - 1,J,K)) {// ingb=10
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 10 * DOF2);
		}
		if (CHECK_BDR(I - 1,J,K + 1)) {// ingb=11
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 11 * DOF2);
		}
		if (CHECK_BDR(I,J,K - 1)) {// ingb=12
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 12 * DOF2);
		}
		if (CHECK_BDR(I,J,K)) {// ingb=13
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.4218750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 13 * DOF2);
		}
		if (CHECK_BDR(I,J,K + 1)) {// ingb=14
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 14 * DOF2);
		}
		if (CHECK_BDR(I + 1,J,K - 1)) {// ingb=15
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 15 * DOF2);
		}
		if (CHECK_BDR(I + 1,J,K)) {// ingb=16
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.4218750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 16 * DOF2);
		}
		if (CHECK_BDR(I + 1,J,K + 1)) {// ingb=17
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,-1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 17 * DOF2);
		}
		if (CHECK_BDR(I - 1,J + 1,K - 1)) {// ingb=18
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0156250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 18 * DOF2);
		}
		if (CHECK_BDR(I - 1,J + 1,K)) {// ingb=19
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 19 * DOF2);
		}
		if (CHECK_BDR(I - 1,J + 1,K + 1)) {// ingb=20
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 20 * DOF2);
		}
		if (CHECK_BDR(I,J + 1,K - 1)) {// ingb=21
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 21 * DOF2);
		}
		if (CHECK_BDR(I,J + 1,K)) {// ingb=22
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.4218750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 22 * DOF2);
		}
		if (CHECK_BDR(I,J + 1,K + 1)) {// ingb=23
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(-1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,0));
				tmp.mla(0.4218750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 23 * DOF2);
		}
		if (CHECK_BDR(I + 1,J + 1,K - 1)) {// ingb=24
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0156250, ptr + SHIFT(1,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,-1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,-1));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 24 * DOF2);
		}
		if (CHECK_BDR(I + 1,J + 1,K)) {// ingb=25
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.4218750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.4218750, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 25 * DOF2);
		}
		if (CHECK_BDR(I + 1,J + 1,K + 1)) {// ingb=26
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K - 1)];
				tmp.mla(0.0156250, ptr + SHIFT(1,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K - 1)];
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,2*K)];
				tmp.mla(0.0468750, ptr + SHIFT(1,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K - 1)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K - 1)];
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,1));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I - 1,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(1,0,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J - 1,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,2*K)];
				tmp.mla(0.1406250, ptr + SHIFT(1,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,1,0));
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I,2*J,2*K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,2*K)];
				tmp.mla(0.4218750, ptr + SHIFT(1,1,1));
				tmp.mla(0.1406250, ptr + SHIFT(1,1,0));
				tmp.mla(0.1406250, ptr + SHIFT(1,0,1));
				tmp.mla(0.0468750, ptr + SHIFT(1,0,0));
				tmp.mla(0.1406250, ptr + SHIFT(0,1,1));
				tmp.mla(0.0468750, ptr + SHIFT(0,1,0));
				tmp.mla(0.0468750, ptr + SHIFT(0,0,1));
				tmp.mla(0.0156250, ptr + SHIFT(0,0,0));
				res += tmp * 0.1250000;
			}
			res.store(AC[C_IDX(I,J,K)] + 26 * DOF2);
		}

	}


#undef C_IDX
#undef F_IDX
#undef SHIFT
}

template<typename idx_t, typename data_t, int dof>
void RAP_3d27_semiXY(
    // 细网格的数据，以及各维度的存储大小（含halo区）
    const data_t * fine_mat, const idx_t fx, const idx_t fy, const idx_t fz, 
    // 粗网格的数据，以及各维度的存储大小（含halo区）
        data_t * coar_mat, const idx_t cx, const idx_t cy, const idx_t cz,
    // 本进程在粗网格上实际负责的范围：[cibeg, ciend) x [cjbeg, cjend) x [ckbeg, ckend)
    const idx_t cibeg, const idx_t ciend, const idx_t cjbeg, const idx_t cjend, const idx_t ckbeg, const idx_t ckend,
    // ilb, iub 分别记录本进程在i维度是否在左边界和右边界
    const bool ilb , const bool iub , 
    // jlb, jub 分别记录本进程在j维度是否在前边界和后边界
    const bool jlb , const bool jub , 
    // klb, kub 分别记录本进程在k维度是否在下边界和上边界
    const bool klb , const bool kub ,
    // 实际就是halo区的宽度，要求粗、细网格的宽度相同
    const idx_t base_x, const idx_t base_y, const idx_t base_z) 
{
	const int DOF2 = dof * dof;
    assert(base_x == 1 && base_y == 1 && base_z == 1);
    for (idx_t i = 0; i < cx * cy * cz * NUM_DIAG * DOF2; i++) {
        // 初值赋为0，为了边界条件
        coar_mat[i] = 0.0;
    }
	
    const data_t (*AF)[NUM_DIAG *DOF2] = (data_t (*)[NUM_DIAG *DOF2])fine_mat;
    data_t (*AC)[NUM_DIAG *DOF2] = (data_t (*)[NUM_DIAG *DOF2])coar_mat;

    const idx_t czcx = cz * cx;// cz * cx
    const idx_t fzfx = fz * fx;// fz * fx
#define C_IDX(i, j, k)   (k) + (i) * cz + (j) * czcx
#define F_IDX(i, j, k)   (k) + (i) * fz + (j) * fzfx
#define SHIFT(oi, oj, ok) (13 + (ok) + (oi) * 3 + (oj) * 9) * DOF2 // 最内维为自由度维
// 当本进程不在某维度的边界时，可以直接算（前提要求细网格的halo区已填充，但R采用4点时似乎也不用？）
#define CHECK_BDR(CI, CJ, CK) \
    (ilb==false || (CI) >= cibeg) && (iub==false || (CI) < ciend) && \
    (jlb==false || (CJ) >= cjbeg) && (jub==false || (CJ) < cjend) && \
    (klb==false || (CK) >= ckbeg) && (kub==false || (CK) < ckend)

	#pragma omp parallel for collapse(3) schedule(static)
    for (idx_t J = cjbeg; J < cjend; J++)
    for (idx_t I = cibeg; I < ciend; I++)
    for (idx_t K = ckbeg; K < ckend; K++) {
		if (CHECK_BDR(I - 1,J - 1,K - 1)) {// ingb=0
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,-1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 0 * DOF2);
		}
		if (CHECK_BDR(I - 1,J - 1,K)) {// ingb=1
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(0,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,0));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 1 * DOF2);
		}
		if (CHECK_BDR(I - 1,J - 1,K + 1)) {// ingb=2
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(0,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 2 * DOF2);
		}
		if (CHECK_BDR(I,J - 1,K - 1)) {// ingb=3
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 3 * DOF2);
		}
		if (CHECK_BDR(I,J - 1,K)) {// ingb=4
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,0));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 4 * DOF2);
		}
		if (CHECK_BDR(I,J - 1,K + 1)) {// ingb=5
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 5 * DOF2);
		}
		if (CHECK_BDR(I + 1,J - 1,K - 1)) {// ingb=6
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,-1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 6 * DOF2);
		}
		if (CHECK_BDR(I + 1,J - 1,K)) {// ingb=7
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(0,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,0));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 7 * DOF2);
		}
		if (CHECK_BDR(I + 1,J - 1,K + 1)) {// ingb=8
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(0,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 8 * DOF2);
		}
		if (CHECK_BDR(I - 1,J,K - 1)) {// ingb=9
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 9 * DOF2);
		}
		if (CHECK_BDR(I - 1,J,K)) {// ingb=10
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,0,0));
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,0));
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(0,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,0));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 10 * DOF2);
		}
		if (CHECK_BDR(I - 1,J,K + 1)) {// ingb=11
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,0,1));
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(0,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 11 * DOF2);
		}
		if (CHECK_BDR(I,J,K - 1)) {// ingb=12
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.5625000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(1,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(0,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(1,0,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(0,0,-1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 12 * DOF2);
		}
		if (CHECK_BDR(I,J,K)) {// ingb=13
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.5625000, ptr + SHIFT(0,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.5625000, ptr + SHIFT(0,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.5625000, ptr + SHIFT(1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.5625000, ptr + SHIFT(1,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,0));
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,0));
				tmp.mla(0.5625000, ptr + SHIFT(0,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.5625000, ptr + SHIFT(0,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,0));
				tmp.mla(0.5625000, ptr + SHIFT(0,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,0));
				tmp.mla(0.5625000, ptr + SHIFT(1,0,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,0));
				tmp.mla(0.5625000, ptr + SHIFT(0,0,0));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 13 * DOF2);
		}
		if (CHECK_BDR(I,J,K + 1)) {// ingb=14
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.5625000, ptr + SHIFT(0,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.5625000, ptr + SHIFT(0,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.5625000, ptr + SHIFT(1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.5625000, ptr + SHIFT(1,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,1));
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,1));
				tmp.mla(0.5625000, ptr + SHIFT(0,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.5625000, ptr + SHIFT(0,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,1));
				tmp.mla(0.5625000, ptr + SHIFT(0,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,1));
				tmp.mla(0.5625000, ptr + SHIFT(1,0,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(-1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,1));
				tmp.mla(0.5625000, ptr + SHIFT(0,0,1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 14 * DOF2);
		}
		if (CHECK_BDR(I + 1,J,K - 1)) {// ingb=15
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.5625000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.5625000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,-1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 15 * DOF2);
		}
		if (CHECK_BDR(I + 1,J,K)) {// ingb=16
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,0,0));
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.5625000, ptr + SHIFT(1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.5625000, ptr + SHIFT(1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,0));
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,0));
				tmp.mla(0.5625000, ptr + SHIFT(1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,0));
				tmp.mla(0.0625000, ptr + SHIFT(0,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,0));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 16 * DOF2);
		}
		if (CHECK_BDR(I + 1,J,K + 1)) {// ingb=17
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,0,1));
				tmp.mla(0.0625000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.5625000, ptr + SHIFT(1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.5625000, ptr + SHIFT(1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,1));
				tmp.mla(0.0625000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(1,-1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,1));
				tmp.mla(0.5625000, ptr + SHIFT(1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,-1,1));
				tmp.mla(0.0625000, ptr + SHIFT(0,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 17 * DOF2);
		}
		if (CHECK_BDR(I - 1,J + 1,K - 1)) {// ingb=18
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,-1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 18 * DOF2);
		}
		if (CHECK_BDR(I - 1,J + 1,K)) {// ingb=19
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(0,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,1,0));
				tmp.mla(0.0625000, ptr + SHIFT(0,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,0));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 19 * DOF2);
		}
		if (CHECK_BDR(I - 1,J + 1,K + 1)) {// ingb=20
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(0,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,1,1));
				tmp.mla(0.0625000, ptr + SHIFT(0,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 20 * DOF2);
		}
		if (CHECK_BDR(I,J + 1,K - 1)) {// ingb=21
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.5625000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,-1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 21 * DOF2);
		}
		if (CHECK_BDR(I,J + 1,K)) {// ingb=22
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,1,0));
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0625000, ptr + SHIFT(1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(0,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,0));
				tmp.mla(0.0625000, ptr + SHIFT(1,0,0));
				tmp.mla(0.5625000, ptr + SHIFT(0,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,0));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 22 * DOF2);
		}
		if (CHECK_BDR(I,J + 1,K + 1)) {// ingb=23
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(0,1,1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0625000, ptr + SHIFT(1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(0,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.0625000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(-1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(-1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,1,1));
				tmp.mla(0.0625000, ptr + SHIFT(1,0,1));
				tmp.mla(0.5625000, ptr + SHIFT(0,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,0,1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 23 * DOF2);
		}
		if (CHECK_BDR(I + 1,J + 1,K - 1)) {// ingb=24
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(1,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(0,1,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(1,0,-1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(1,1,-1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,-1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,-1));
				tmp.mla(0.0625000, ptr + SHIFT(0,0,-1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 24 * DOF2);
		}
		if (CHECK_BDR(I + 1,J + 1,K)) {// ingb=25
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(1,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,1,0));
				tmp.mla(0.0625000, ptr + SHIFT(0,1,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,1,0));
				tmp.mla(0.0625000, ptr + SHIFT(1,0,0));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(1,1,0));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,0));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,0));
				tmp.mla(0.0625000, ptr + SHIFT(0,0,0));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 25 * DOF2);
		}
		if (CHECK_BDR(I + 1,J + 1,K + 1)) {// ingb=26
			PointerWrapper<DOF2, data_t> res(0.0);
			{// u_coord=(2*I - 1,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J - 1,K)];
				tmp.mla(0.0625000, ptr + SHIFT(1,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J - 1,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J - 1,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,1,1));
				tmp.mla(0.0625000, ptr + SHIFT(0,1,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - 1,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I - 1,2*J,K)];
				tmp.mla(0.1875000, ptr + SHIFT(1,1,1));
				tmp.mla(0.0625000, ptr + SHIFT(1,0,1));
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I,2*J,K)
				PointerWrapper<DOF2, data_t> tmp(0.0);
				const data_t * ptr = AF[F_IDX(2*I,2*J,K)];
				tmp.mla(0.5625000, ptr + SHIFT(1,1,1));
				tmp.mla(0.1875000, ptr + SHIFT(1,0,1));
				tmp.mla(0.1875000, ptr + SHIFT(0,1,1));
				tmp.mla(0.0625000, ptr + SHIFT(0,0,1));
				res += tmp * 0.2500000;
			}
			res.store(AC[C_IDX(I,J,K)] + 26 * DOF2);
		}
	}


#undef C_IDX
#undef F_IDX
#undef SHIFT
}


#undef NUM_DIAG

#endif