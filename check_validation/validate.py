import sys
import os
from Bio.Align import PairwiseAligner
from Bio import SeqIO
import glob


def read_fasta(filepath):
    try:
        record = next(SeqIO.parse(filepath, "fasta"))
        return str(record.seq)
    except (FileNotFoundError, StopIteration):
        print(f"Failed to read: {filepath}")
        return None


def parse_alignment(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        score_line = [line for line in content.split('\n') if 'Alignment Score:' in line]
        if not score_line:
            return None, None, None
            
        score = int(score_line[0].split(':')[1].strip())
        lines = content.split('\n')
        seq_a, seq_b = None, None
        
        for i, line in enumerate(lines):
            if ('Aligned Human:' in line or 'Aligned A:' in line) and i + 1 < len(lines):
                seq_a = lines[i + 1].strip()
            elif ('chimp' in line.lower() or 'gorilla' in line.lower() or 'mouse' in line.lower() or 
                  'cow' in line.lower() or 'pig' in line.lower() or 'dog' in line.lower() or
                  'orangutan' in line.lower() or 'bonobo' in line.lower() or 'rhmonkey' in line.lower() or
                  'gibbon' in line.lower() or 'bluewhale' in line.lower() or 'tupaiachinensis' in line.lower() 
                  or 'Aligned B:' in line) and i + 1 < len(lines):
                seq_b = lines[i + 1].strip()
        
        return seq_a, seq_b, score if seq_a and seq_b else (None, None, None)
    except FileNotFoundError:
        return None, None, None


def strip_gaps(seq_a, seq_b):
    return seq_a.replace('_', ''), seq_b.replace('_', '')


def validate_with_biopython(orig_a, orig_b, c_score, species, ref_seq):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.gap_score = -1
    
    ref_match = orig_a == ref_seq
    alignments = aligner.align(orig_a, orig_b)
    bio_score = int(alignments[0].score)
    score_match = abs(c_score - bio_score) < 0.001
    
    print(f"[{species}]")
    print(f"  Reference match: {'✓' if ref_match else '✗'}")
    if not ref_match:
        print(f"    Extracted: {len(orig_a)}, Reference: {len(ref_seq)}")
    print(f"  C score: {c_score}")
    print(f"  BioPython: {bio_score}")
    print(f"  Score match: {'✓' if score_match else '✗'}")
    
    if not score_match:
        print(f"  WARNING: Score mismatch!")
    
    return score_match, bio_score, ref_match


def load_mrna():
    path = ""
    seqs = {}
    
    files = glob.glob(os.path.join(path, "*_BRCA1_mRNA.fasta"))
    print(f"Found {len(files)} mRNA files")
    
    for f in files:
        name = os.path.basename(f).replace("_BRCA1_mRNA.fasta", "")
        seq = read_fasta(f)
        if seq:
            seqs[name] = seq
            print(f"  {name}: {len(seq)} bp")
    
    return seqs
def load_mito():
    path = ""
    seqs = {}
    
    files = glob.glob(os.path.join(path, "*_mitochondrion.fasta"))
    print(f"Found {len(files)} mitochondrial files")
    
    mapping = {
        'Homosapiens': 'human',
        'Gorilla': 'Gorilla', 
        'Pantroglodytes': 'Chimpanzee',
        'Pongopygmaeus': 'Orangutan',
        'Panpaniscus': 'Bonobo',
        'Hylobateslar': 'Gibbon',
        'Musmusculus': 'Mouse',
        'Canislupus': 'Dog',
        'Bostaurus': 'Cow',
        'Balaenopteramusculus': 'BlueWhale',
        'tupaiachinensis': 'TreeShrew'
    }
    
    for f in files:
        prefix = os.path.basename(f).replace("_mitochondrion.fasta", "")
        name = mapping.get(prefix, prefix)
        seq = read_fasta(f)
        if seq:
            seqs[name] = seq
            print(f"  {name}: {len(seq)} bp")
    
    return seqs


def find_results():
    files = []
    for pattern in ["human_*_alignment.txt", "human_*_mito_alignment.txt"]:
        for f in glob.glob(pattern):
            if "_mito_alignment.txt" in f:
                species = f.replace("human_", "").replace("_mito_alignment.txt", "")
            else:
                species = f.replace("human_", "").replace("_alignment.txt", "")
            files.append((species, f))
    return files


def validate_results(seqs, files, ref):
    total = score_ok = ref_ok = 0
    
    for species, filename in files:
        seq_a, seq_b, c_score = parse_alignment(filename)
        if not all([seq_a, seq_b, c_score is not None]):
            print(f"[{species}] Failed to read: {filename}")
            continue
            
        orig_a, orig_b = strip_gaps(seq_a, seq_b)
        
        if species not in seqs:
            print(f"[{species}] No reference sequence")
            continue
        
        try:
            score_match, bio_score, ref_match = validate_with_biopython(
                orig_a, orig_b, c_score, species, ref
            )
            
            total += 1
            if score_match: score_ok += 1
            if ref_match: ref_ok += 1
                
        except Exception as e:
            print(f"[{species}] Error: {e}")
        
        print("-" * 40)
    
    return total, score_ok, ref_ok

def print_summary(total, score_ok, ref_ok):
    print("\n=== Summary ===")
    print(f"Total tests: {total}")
    print(f"Score matches: {score_ok}/{total}")
    print(f"Reference matches: {ref_ok}/{total}")
    
    if total > 0:
        print(f"Score accuracy: {score_ok/total*100:.1f}%")
        print(f"Reference accuracy: {ref_ok/total*100:.1f}%")
        
        if score_ok == total and ref_ok == total:
            print("\n✓ All tests passed!")
        elif score_ok == total:
            print(f"\n⚠ Scores match but {total-ref_ok} reference mismatches")
        else:
            print(f"\n✗ {total-score_ok} score mismatches, {total-ref_ok} reference mismatches")

def validate_mrna():
    print("=== BRCA1 mRNA Validation ===\n")
    
    seqs = load_mrna()
    if 'human' not in seqs:
        print("No human reference!")
        return
        
    human_ref = seqs['human']
    print(f"\nHuman reference loaded: {len(human_ref)} bp\n")
    
    files = find_results()
    valid_files = [(s, f) for s, f in files if s in seqs and "_mito_" not in f]
    
    if not valid_files:
        print("No alignment files found!")
        return
    
    print(f"Found {len(valid_files)} files to validate\n")
    total, score_ok, ref_ok = validate_results(seqs, valid_files, human_ref)
    print_summary(total, score_ok, ref_ok)

def validate_mito():
    print("=== Mitochondrial Genome Validation ===\n")
    
    seqs = load_mito()
    if 'human' not in seqs:
        print("No human mitochondrial reference!")
        return
        
    human_ref = seqs['human']
    print(f"\nHuman mitochondrial reference: {len(human_ref)} bp\n")

    c_path = ""
    files = []
    for f in glob.glob(os.path.join(c_path, "human_*_mito_alignment.txt")):
        name = os.path.basename(f).replace("human_", "").replace("_mito_alignment.txt", "")
        files.append((name, f))
    
    valid_files = [(s, f) for s, f in files if s in seqs]
    
    if not valid_files:
        print("No mitochondrial alignment files found!")
        return
    
    print(f"Found {len(valid_files)} files to validate\n")
    total, score_ok, ref_ok = validate_results(seqs, valid_files, human_ref)
    print_summary(total, score_ok, ref_ok)

def quick_align():
    print("\n=== Reference Alignment Check ===")
    
    seqs = load_mrna()
    if 'human' not in seqs:
        print("No human reference")
        return
        
    human_seq = seqs['human']
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.gap_score = -1
    
    print(f"Reference length: {len(human_seq)} bp\n")
    
    for name, seq in seqs.items():
        if name == 'human':
            continue
            
        print(f"[{name}]")
        print(f"  Length: {len(seq)} bp")
        
        try:
            alignment = aligner.align(human_seq, seq)[0]
            score = int(alignment.score)
            
            print(f"  Score: {score}")
            print(f"  Length: {alignment.shape[1]}")
            
            h_align = str(alignment[0])
            s_align = str(alignment[1])
            
            matches = sum(1 for i in range(len(h_align)) 
                         if h_align[i] != '-' and s_align[i] != '-' 
                         and h_align[i] == s_align[i])
            gaps = h_align.count('-') + s_align.count('-')
            total_pos = len(h_align)
            
            print(f"  Matches: {matches}, Gaps: {gaps}, Total: {total_pos}")
            print(f"  Identity: {matches/(total_pos-gaps)*100:.1f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mito":
        validate_mito()
    else:
        validate_mrna()
        quick_align()
        print("\n" + "="*60)
        validate_mito()
