#!/usr/bin/env python3
"""
generate_barcodes.py
====================
Generate a library of orthogonal 25-nt DNA barcodes for Sidewinder junction
addressing using the seqwalk package (https://github.com/ggdna/seqwalk).

Algorithm overview
------------------
1. Call seqwalk.design.max_orthogonality to build a de-Bruijn-graph library
   whose SSM (Sequence Symmetry Minimisation) k value is automatically chosen
   as the highest k that still yields ≥ N_DESIGN sequences satisfying:
     • 4-letter alphabet (ACGT)
     • length 25 nt
     • GC content 40–60 % (10–15 GC bases out of 25)
     • no run of ≥ 5 identical bases  (prevented_patterns)
     • reverse-complement pairs excluded (RCfree=True)
2. Apply post-filters for tandem repeats (period 2–4, total length ≥ 8 nt).
3. Validate ≥ N_TARGET barcodes remain; raise a descriptive error if not.
4. Write CSV: junction_id | barcode_forward | barcode_reverse_complement

Dependencies
------------
    pip install seqwalk          (installs seqwalk + numpy)

All other imports are from the Python standard library.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Configurable parameters  ← tweak these to adjust the design
# ─────────────────────────────────────────────────────────────────────────────

# Minimum number of barcode pairs required in the final CSV.
N_TARGET = 1_200

# Number of sequences requested from seqwalk.
# Oversampling above N_TARGET leaves headroom for post-filters.
# Increase this if the assertion at the end fails.
N_DESIGN = 1_800

# Fixed barcode length (nt).
BARCODE_LENGTH = 25

# GC content limits expressed as base *counts* (not fractions) out of BARCODE_LENGTH.
# 10/25 = 40 %, 15/25 = 60 %.
GC_MIN = 10
GC_MAX = 15

# Reject sequences with any run of *more* than this many identical bases.
# Example: MAX_HOMOPOLYMER = 4 → reject "AAAAA" (run of 5) but allow "AAAA".
MAX_HOMOPOLYMER = 4

# Reject tandem repeats (period 2–4 nt) whose total length reaches this threshold.
# Example: MIN_REPEAT_LEN = 8 → reject "ATATATAT" (AT × 4 = 8 nt).
MIN_REPEAT_LEN = 8

# Output file name.
OUTPUT_FILE = "sidewinder_25nt_barcodes.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import sys
import re
import csv
import statistics

# Install seqwalk automatically if it isn't present.
try:
    from seqwalk import design
except ImportError:
    import subprocess
    print("seqwalk not found — installing via pip …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seqwalk"])
    from seqwalk import design

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Generate the initial library with seqwalk
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Step 1 – seqwalk.design.max_orthogonality")
print("=" * 65)
print(f"  Requesting  : {N_DESIGN:,} sequences, length {BARCODE_LENGTH} nt")
print(f"  GC limits   : {GC_MIN}–{GC_MAX} bases ({100*GC_MIN/BARCODE_LENGTH:.0f}–"
      f"{100*GC_MAX/BARCODE_LENGTH:.0f} %)")
print(f"  RCfree      : True  (rev-comp pairs excluded from library)")
print(f"  Alphabet    : ACGT")
print()

# seqwalk sweeps k from high to low and picks the *largest* k (i.e., the
# most stringent SSM constraint) that still produces ≥ N_DESIGN sequences.
# A higher k means any k-mer substring occurs at most once across ALL sequences
# in the library, providing stronger cross-hybridisation suppression.
#
# prevented_patterns: using 5-mers (AAAAA …) matches the user's requirement of
# "no run of MORE than 4 identical bases".  The seqwalk default uses 4-mers
# (AAAA …) which is stricter; we relax to 5-mers to maximise library size
# while still satisfying the specification.
raw_library = design.max_orthogonality(
    N_DESIGN,
    BARCODE_LENGTH,
    alphabet="ACGT",
    RCfree=True,
    GClims=(GC_MIN, GC_MAX),
    prevented_patterns=["AAAAA", "CCCCC", "GGGGG", "TTTTT"],
    verbose=True,
)

n_raw = len(raw_library)
print(f"\n  seqwalk returned : {n_raw:,} sequences")

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def gc_count(seq: str) -> int:
    """Return the number of G/C bases in *seq*."""
    return sum(1 for b in seq if b in "GC")


def has_long_homopolymer(seq: str, max_run: int = MAX_HOMOPOLYMER) -> bool:
    """Return True if *seq* contains a run of more than *max_run* identical bases.

    This is a safety net; seqwalk's prevented_patterns already blocks runs of
    length ≥ len(prevented_pattern), so this filter should rarely trigger.
    """
    # Regex: single char captured, repeated max_run or more additional times
    # → total run length > max_run.
    pattern = r"(.)\1{" + str(max_run) + r",}"
    return bool(re.search(pattern, seq))


def has_tandem_repeat(seq: str, min_total: int = MIN_REPEAT_LEN) -> bool:
    """Return True if *seq* contains a tandem repeat of total length ≥ *min_total*.

    Checks repeat periods 2–4 nt.  Period-1 (homopolymers) is handled by
    has_long_homopolymer(); starting at period 2 avoids double-counting.

    Example (min_total=8):
        period=2 → rejects "ATATATAT"  (AT × 4 = 8 nt)
        period=3 → rejects "ATCATCATC" (ATC × 3 = 9 nt)
        period=4 → rejects "ACGTACGT"  (ACGT × 2 = 8 nt)
    """
    for period in range(2, 5):
        # Compute how many *additional* repetitions of the captured group are
        # needed so that total length (1 capture + N repeats) × period ≥ min_total.
        # total_reps = ceil(min_total / period), additional = total_reps - 1.
        total_reps = -(-min_total // period)          # ceiling division
        additional = total_reps - 1
        if additional < 1:
            additional = 1
        pattern = r"(.{" + str(period) + r"})\1{" + str(additional) + r",}"
        if re.search(pattern, seq):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Post-filters
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Step 2 – Post-filters")
print("=" * 65)

# 2a. GC content  (seqwalk enforces GClims internally, but double-check here)
after_gc = [s for s in raw_library if GC_MIN <= gc_count(s) <= GC_MAX]
n_dropped_gc = n_raw - len(after_gc)
print(f"  After GC filter [{GC_MIN}–{GC_MAX} bases]  : "
      f"{len(after_gc):>6,}  (dropped {n_dropped_gc:,})")

# 2b. Homopolymers  (> MAX_HOMOPOLYMER consecutive identical bases)
after_hp = [s for s in after_gc if not has_long_homopolymer(s)]
n_dropped_hp = len(after_gc) - len(after_hp)
print(f"  After homopolymer filter [run > {MAX_HOMOPOLYMER}] : "
      f"{len(after_hp):>6,}  (dropped {n_dropped_hp:,})")

# 2c. Tandem repeats  (period 2–4, total length ≥ MIN_REPEAT_LEN)
after_rep = [s for s in after_hp if not has_tandem_repeat(s)]
n_dropped_rep = len(after_hp) - len(after_rep)
print(f"  After tandem-repeat filter [≥{MIN_REPEAT_LEN} nt] : "
      f"{len(after_rep):>6,}  (dropped {n_dropped_rep:,})")

final_library = after_rep

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Validate minimum library size
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Step 3 – Validation")
print("=" * 65)

if len(final_library) < N_TARGET:
    raise RuntimeError(
        f"\n[ERROR] Only {len(final_library):,} barcodes survived all filters "
        f"(need ≥ {N_TARGET:,}).\n"
        "Suggested remedies (in order of preference):\n"
        "  1. Increase N_DESIGN (e.g., 2500) to request more sequences.\n"
        "  2. Widen GC limits, e.g., GC_MIN=9, GC_MAX=16.\n"
        "  3. Raise MAX_HOMOPOLYMER to 5 or increase MIN_REPEAT_LEN to 10.\n"
        "  4. Switch to design.max_size with a lower k value for a larger library."
    )

print(f"  Final library size : {len(final_library):,}  (≥ {N_TARGET:,} required ✓)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – GC content summary
# ─────────────────────────────────────────────────────────────────────────────
gc_counts = [gc_count(s) for s in final_library]
gc_pct    = [100.0 * g / BARCODE_LENGTH for g in gc_counts]

print()
print(f"  GC content summary (n = {len(final_library):,}):")
print(f"    Min  : {min(gc_counts):2d} bases  ({min(gc_pct):.1f} %)")
print(f"    Max  : {max(gc_counts):2d} bases  ({max(gc_pct):.1f} %)")
print(f"    Mean : {statistics.mean(gc_counts):.2f} bases  ({statistics.mean(gc_pct):.1f} %)")
print(f"    SD   : {statistics.stdev(gc_counts):.2f} bases")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Compute reverse complements
# ─────────────────────────────────────────────────────────────────────────────

# Translation table for Watson-Crick complementation.
_COMPLEMENT_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    """Return the reverse Watson-Crick complement of *seq*."""
    return seq.translate(_COMPLEMENT_TABLE)[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Write CSV
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Step 6 – Write CSV")
print("=" * 65)

with open(OUTPUT_FILE, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["junction_id", "barcode_forward", "barcode_reverse_complement"])
    for i, seq in enumerate(final_library, start=1):
        writer.writerow([i, seq, reverse_complement(seq)])

print(f"  Wrote {len(final_library):,} barcode pairs → '{OUTPUT_FILE}'")
print()
print("Done.")
print("=" * 65)
