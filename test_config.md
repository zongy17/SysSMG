executable = "syssmg_All64.exe" and "syssmg_K64P32D16.exe" on ARM-based platform, while only "syssmg_All64.exe" on X86 platform.

# Rhd-3T
case_name = "LASER"
## Case: size of 128x128x128
cx = 128, cy = 128, cz = 128
| Node | Rankfile | px | py | pz | other_param |
|------|:--------:|---:|----|----|-------------|
|  1   | N0.5P64  | 4  | 4  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
|  1   |  N1P64   | 4  | 4  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
|  2   |  N2P64   | 4  | 4  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
|  4   |  N4P512  |  8 | 8  |  8 |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
|  8   |  N8P512  |  8 | 8  |  8 |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
| 16   | N16P512  |  8 | 8  |  8 |CG 10 GMG 0 4 PGS PGS PGS PGS LU |

This is for strong scalability tests.
# Solid-3D
case_name = "SOLID"
## Case 1: 80x96x64
cx = 80, cy = 96, cz = 64
| Node | Rankfile | px | py | pz | other_param |
|------|:--------:|---:|----|----|-------------|
|  1   | N0.5P60  | 5  | 3  | 2  |CG 10 GMG 0 3 PGS PGS PGS PGS|
|  1   |  N1P60   | 5  | 3  | 2  |CG 10 GMG 0 3 PGS PGS PGS PGS|
|  1   |  N1P120  | 5  | 3  | 4  |CG 10 GMG 0 3 PGS PGS PGS PGS|

This is for weak scalability tests.
## Case 2: 80x96x128
cx = 80, cy = 96, cz = 128
| Node | Rankfile | px | py | pz | other_param |
|------|:--------:|---:|----|----|-------------|
|  1   |  N1P60   | 5  | 6  | 2  |CG 10 GMG 0 3 PGS PGS PGS PGS|
|  1   |  N1P60   | 5  | 6  | 2  |CG 10 GMG 0 3 PGS PGS PGS PGS|
|  1   |  N1P120  | 5  | 6  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU|
|  2   |  N2P60   | 5  | 6  | 2  |CG 10 GMG 0 3 PGS PGS PGS PGS|
|  2   |  N2P60   | 5  | 6  | 2  |CG 10 GMG 0 3 PGS PGS PGS PGS|
|  2   |  N2P120  | 5  | 6  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU|

This is for weak scalability tests.
## Case 3: 160x96x128
cx = 160, cy = 96, cz = 128
| Node | Rankfile | px | py | pz | other_param |
|------|:--------:|---:|----|----|-------------|
|  2   |  N1P24   | 1  | 6  | 4  |CG 10 GMG 0 3 PGS PGS PGS PGS|
|  2   |  N1P24   | 1  | 6  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU|
|  2   |  N1P120  | 5  | 6  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU|
|  4   |  N4P48   | 1  | 6  | 8  |CG 10 GMG 0 4 PGS PGS PGS PGS LU|
|  4   |  N4P48   | 2  | 6  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU|
|  4   |  N4P120  | 5  | 6  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU|

This is for weak scalability tests.

## Case 4: 160x192x128
cx = 160, cy = 192, cz = 128
| Node | Rankfile | px | py | pz | other_param |
|------|:--------:|---:|----|----|-------------|
|  1   |  N1P120  | 5  | 6  | 4  |CG 10 GMG 1 3 PGS PGS PGS PGS PGS|
|  2   |  N1P120  | 5  | 6  | 4  |CG 10 GMG 1 3 PGS PGS PGS PGS PGS|
|  4   |  N4P120  | 5  | 6  | 4  |CG 10 GMG 1 3 PGS PGS PGS PGS PGS|
|  4   |  N4P120  | 5  | 6  | 4  |CG 10 GMG 1 4 PGS PGS PGS PGS PGS LU|
|  8   |  N8P120  | 5  | 6  | 4  |CG 10 GMG 1 3 PGS PGS PGS PGS LU|
|  8   |  N8P120  | 5  | 6  | 4  |CG 10 GMG 1 4 PGS PGS PGS PGS PGS LU|
| 15   | N15P120  | 5  | 6  | 4  |CG 10 GMG 1 3 PGS PGS PGS PGS PGS|
| 15   | N15P120  | 5  | 6  | 4  |CG 10 GMG 1 4 PGS PGS PGS PGS PGS LU|
| 16   | N16P240  | 5  | 6  | 8  |CG 10 GMG 1 4 PGS PGS PGS PGS PGS LU|
| 16   | N16P240  | 5  | 6  | 8  |CG 10 GMG 0 4 PGS PGS PGS PGS LU|
| 30   | N30P120  | 5  | 6  | 4  |CG 10 GMG 1 4 PGS PGS PGS PGS PGS LU|
| 30   | N30P120  | 5  | 6  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU|

This is for strong and weak scalability tests.
## Case 5: 160x192x256
cx = 160, cy = 192, cz = 256
| Node | Rankfile | px | py | pz | other_param |
|------|:--------:|---:|----|----|-------------|
|  8   |  N8P120  | 5  | 6  | 4  |CG 10 GMG 0 5 PGS PGS PGS PGS PGS LU|
|  8   |  N8P240  | 5  | 6  | 8  |CG 10 GMG 0 5 PGS PGS PGS PGS PGS LU|
| 15   | N15P120  | 5  | 6  | 4  |CG 10 GMG 0 5 PGS PGS PGS PGS PGS LU|

This is for weak scalability tests.
