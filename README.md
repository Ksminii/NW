# NW

## Directory Structure

```
NW/
├── Basic_implementations/          # Basic NW algorithm implementations
│   ├── nw_linear.c                # C implementation (linear gap)
│   ├── nw_affine.c                # C implementation (affine gap)
│   └── nw_linear.py               # Python implementation (linear gap)
├── Accerlerated_implementations/   # Optimized/accelerated versions
├── check_validation/               # Validation tools
│   └── validate.py                # Validate alignment results with BioPython
└── README.md
```

## Dataset Download

Download genomic sequences from [NCBI](https://www.ncbi.nlm.nih.gov/nucleotide/) in FASTA format.

### BRCA1 mRNA
Search: `"BRCA1"[Gene] AND "species"[Organism] AND mRNA[Filter]`

Species: Human, Chimpanzee, Gorilla, Orangutan, Bonobo, Gibbon, Mouse, Dog, Cow

### Mitochondrial Genomes
Search: `"species"[Organism] AND mitochondrion[Filter] AND complete genome`

Species: Human, Chimpanzee, Gorilla, Orangutan, Bonobo, Gibbon, Mouse, Dog, Cow, Blue Whale, Tree Shrew

## Usage

### Compile and Run (C)
```bash
cd Basic_implementations
gcc -O3 nw_linear.c -o nw_linear
./nw_linear seq1.fasta seq2.fasta

gcc -O3 nw_affine.c -o nw_affine
./nw_affine seq1.fasta seq2.fasta
```

### Run (Python)
```bash
cd Basic_implementations
python3 nw_linear.py seq1.fasta seq2.fasta
```

### Validate Results
```bash
cd check_validation
python3 validate.py
```
