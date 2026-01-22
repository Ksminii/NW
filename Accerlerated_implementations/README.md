### Overview

```
[Host (CPU)]                                   [Device (GPU)]
    |
    ├─ Read FASTA 
    ├─ Memory allocation (DP matrix, sequences)
    ├─ Data transfer      ────────────────────→  GPU Memory
    |
    ├─ Wavefront Loop 
    |     for k = 1 to m+n:
    |       launch kernel ──────────────────→  compute_diagonal<<<>>>
    |                                        ├─ compute cellos on same diagonal on parallel
    |                                        └─ Stroe DP + Traceback result
    |       Wait Sync()   ←──────────────────
    |
    ├─ Result ←────────────────────── GPU Memory
    ├─ Traceback (on cpu)
    └─ Write 
```
