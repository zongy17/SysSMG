For all cases, case_name = "OIL" and other_param = "GMRES 10 GMG 0 2 PGS PGS PGS", while cx, cy, cz, and px, py, pz are shown below specifically.

executable = "syssmg_All64.exe" and "syssmg_K64P32D16.exe" on ARM-based platform, while only "syssmg_All64.exe" on X86 platform.

# Case: size of 320x64x384

cx = 320, cy = 64, cz = 384
| Node | Rankfile | px | py | pz |
|------|:--------:|---:|----|----|
| 2    |  N2P240  | 20 | 1  | 12 |
| 2    |  N2P120  | 10 | 1  | 12 |
| 2    |  N2P80   | 10 | 1  |  8 |
| 2    |  N2P48   |  8 | 1  |  6 |
| 2    |  N2P40   |  5 | 1  |  8 |
| 4    |  N4P480  | 20 |  1	| 24 |
| 4    |  N4P240  | 20 |  1 | 12 |
| 4    |  N4P160  | 10 |  1 | 16 |
| 4    |  N4P96   |  8 |  1 | 12 |
| 4    |  N4P80   | 10 |  1 |  8 |
| 8    |  N8P960  | 40 |  1 | 24 |
| 8    |  N8P480  | 20 |  1 | 24 |
| 8    |  N8P320  | 20 |  1 | 16 |
| 8    |  N8P192  | 16 |  1 | 12 |
| 8    |  N8P160  | 10 |  1 | 16 |
| 16   | N16P1920 | 40 |  1 | 48 |
| 16   | N16P960  | 40 |  1 | 24 |
| 16   | N16P640  | 20 |  1 | 32 |
| 16   | N16P384  | 16 |  1 | 24 |
| 16   | N16P320  | 20 |  1 | 16 |
| 32   | N32P3840 | 80 |  1 | 48 |
| 32   | N32P1920 | 40 |  1 | 48 |
| 32   | N32P1280 | 40 |  1 | 32 |
| 32   | N32P768  | 32 |  1 | 24 |
| 32   | N32P640  | 20 |  1 | 32 |
| 64   | N64P7680 | 80 |  1 | 96 |
| 64   | N64P3840 | 80 |  1 | 48 |
| 64   | N64P2560 | 40 |  1 | 64 |
| 64   | N64P1536 | 32 |  1 | 48 |
| 64   | N64P1280 | 40 |  1 | 32 |

This is for strong scalability tests.

