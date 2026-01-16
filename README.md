# NW

  ## Directory Structure

  NW/
  ├── Basic_implementations/          # Basic NW algorithm implementations
  │   ├── nw_linear.c                # C implementation (linear gap)
  │   ├── nw_affine.c                # C implementation (affine gap)
  │   ├── nw_linear.py               # Python implementation (linear gap)
  │   └── hirschberg_generic.c       # space-efficient algorithm
  ├── Accerlerated_implementations/   # GPU/parallel accelerated versions
  │   ├── nw_ocl_generic.c           # OpenCL implementation (general)
  │   └── nw_cuda_generic.cu         # CUDA implementation
  ├── check_validation/               # Validation tools
  │   └── validate.py                # Validate alignment results with BioPython
  └── README.md

  ## Usage

  ### Basic Implementations

  #### C - Linear Gap
  ```bash
  cd Basic_implementations
  gcc -O3 nw_linear.c -o nw_linear
  ./nw_linear seq1.fasta seq2.fasta

  C - Affine Gap

  cd Basic_implementations
  gcc -O3 nw_affine.c -o nw_affine
  ./nw_affine seq1.fasta seq2.fasta

  C - Hirschberg (Space-Efficient)

  cd Basic_implementations
  gcc -O3 hirschberg_generic.c -o hirschberg_generic
  ./hirschberg_generic seq1.fasta seq2.fasta

  Python - Linear Gap

  cd Basic_implementations
  python3 nw_linear.py seq1.fasta seq2.fasta

  Accelerated Implementations

  OpenCL - Generic

  cd Accerlerated_implementations

  # macOS
  gcc -o nw_ocl_generic nw_ocl_generic.c -framework OpenCL

  # Linux
  gcc -o nw_ocl_generic nw_ocl_generic.c -lOpenCL

  # Run
  ./nw_ocl_generic seq1.fasta seq2.fasta

  CUDA

  cd Accerlerated_implementations

  # Compile
  nvcc -O3 nw_cuda_generic.cu -o nw_cuda_generic

  # Run
  ./nw_cuda_generic seq1.fasta seq2.fasta

  Validate Results

  cd check_validation
  python3 validate.py
