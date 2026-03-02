"""
collatz_sequence.py
===================
Implements the review paper's image-dependent dynamic key generation.

Takes a hardcoded private key and combines it with the image hash
using XOR to produce image-dependent chaotic parameters r1, r2, x1, x2.

Private Key  ──────────────────────┐
                                    ▼
Image ──► SIDH_hash ──► hash ──► XOR combine
                                    │
                                    ▼
                             Parse t, p, q, s
                                    │
                                    ▼
                       Run 4 Collatz Sequences
                       from t, t+q, t, t+p
                                    │
                                    ▼
                        Extract LSD digits at
                        indexes p, p-s, q, p+s
                                    │
                                    ▼
                          Map to chaotic range
                                    │
                                    ▼
                            r1, r2, x1, x2
"""

import hash as image_hash


# ═══════════════════════════════════════════════════════════════
# PRIVATE KEY  —  hardcoded dummy key (15 digits as per paper)
# ═══════════════════════════════════════════════════════════════

PRIVATE_KEY = "965412038794564"

#  Structure (Table 4 from paper):
#  Position:  0 1 2 3 4 5 6 | 7 8 9 | 10 11 12 | 13 14
#  Part:          t          |   p   |    q     |   s
#             9654120        | 387   |  945     |  64


# ═══════════════════════════════════════════════════════════════
# FUNCTION 1  —  collatz_sequence(n)
# ═══════════════════════════════════════════════════════════════

def collatz_sequence(n):
    """
    Generate the full Collatz sequence from n down to 1.

    Rule:
        if n is even  →  n = n / 2
        if n is odd   →  n = 3n + 1
    Repeat until n = 1.

    Returns the complete list of numbers in the sequence.
    """
    seq = []
    while n != 1:
        seq.append(n)
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
    seq.append(1)
    return seq


# ═══════════════════════════════════════════════════════════════
# FUNCTION 2  —  combine_key_and_hash(private_key, hex_hash)
# ═══════════════════════════════════════════════════════════════

def combine_key_and_hash(private_key, hex_hash):
    """
    Core of the review paper's contribution.

    XOR the private key with the image hash so that:
        - Same key  + different image  →  different combined value
        - Different key + same image   →  different combined value
        - Both must match to reproduce the same parameters

    Steps:
        1. Convert private key (string of digits) to integer
        2. Convert image hash (hex string)        to integer
        3. XOR them  →  combined integer
        4. Take first 15 decimal digits of result
    """
    key_int  = int(private_key)
    hash_int = int(hex_hash, 16)

    # XOR — any change in image or key flips bits here
    combined_int    = key_int ^ hash_int
    combined_digits = str(combined_int).zfill(15)[:15]

    return combined_digits


# ═══════════════════════════════════════════════════════════════
# FUNCTION 3  —  parse_key_components(combined_digits)
# ═══════════════════════════════════════════════════════════════

def parse_key_components(combined_digits):
    """
    Split 15 combined digits into t, p, q, s
    exactly as Table 4 in the paper:

        t = first 7 digits  →  Collatz seed
        p = next  3 digits  →  page index 1
        q = next  3 digits  →  page index 2
        s = last  2 digits  →  offset
    """
    t = int(combined_digits[0:7])
    p = int(combined_digits[7:10])
    q = int(combined_digits[10:13])
    s = int(combined_digits[13:15])

    # t must be >= 2 for Collatz to run
    if t < 2:
        t = 2

    return t, p, q, s


# ═══════════════════════════════════════════════════════════════
# FUNCTION 4  —  extract_lsd_digits(sequence, start_index)
# ═══════════════════════════════════════════════════════════════

def extract_lsd_digits(sequence, start_index, length=15):
    """
    Algorithm 1 Steps 4-5 / 7-8 / 10-11 / 13-14:

        1. Start at position 'start_index' in the sequence
           (wraps around if index exceeds sequence length)
        2. Take 15 consecutive elements
        3. Extract last digit (mod 10) of each element
        4. Combine into one 15-digit number
        5. Divide by 1e15  →  value in [0, 1)
    """
    # Wrap index safely
    start_index = start_index % len(sequence)

    # Pick 15 consecutive elements
    selected = []
    for i in range(length):
        idx = (start_index + i) % len(sequence)
        selected.append(sequence[idx])

    # Extract last digit of each
    lsd_digits = [n % 10 for n in selected]

    # Build one number from all digits
    combined = 0
    for d in lsd_digits:
        combined = combined * 10 + d

    # Normalise to [0, 1)
    return combined / 1e15


# ═══════════════════════════════════════════════════════════════
# FUNCTION 5  —  map_to_chaotic_range(r_raw)
# ═══════════════════════════════════════════════════════════════

def map_to_chaotic_range(r_raw):
    """
    Algorithm 1, Steps 15-16:

    The logistic map only behaves chaotically when r is in [3.57, 4.00].
    Map the raw value (which is in [0, 1)) into that range:

        r = 3.57 + r_raw * 0.43
    """
    return 3.57 + r_raw * 0.43


# ═══════════════════════════════════════════════════════════════
# FUNCTION 6  —  generate_params(image_path)
# ═══════════════════════════════════════════════════════════════

def generate_params(image_path):
    """
    Main function — full Algorithm 1 pipeline with image-dependent key.

    Returns: r1, r2, x1, x2
    """

    print("=" * 55)
    print("  COLLATZ-BASED CHAOTIC PARAMETER GENERATION")
    print("  (Review Paper: Image-Dependent Dynamic Key)")
    print("=" * 55)

    # ── Stage 1: Get image hash ──────────────────────────────
    print(f"\n[STAGE 1] Hashing image...")
    hex_hash = image_hash.SIDH_hash(image_path)
    print(f"  Image        : {image_path}")
    print(f"  Private Key  : {PRIVATE_KEY}")
    print(f"  Image Hash   : {hex_hash}")

    # ── Stage 2: XOR combine key and hash ───────────────────
    print(f"\n[STAGE 2] XOR combining private key and image hash...")
    combined_digits = combine_key_and_hash(PRIVATE_KEY, hex_hash)
    print(f"  Combined (15 digits) : {combined_digits}")

    # ── Stage 3: Parse into t, p, q, s ──────────────────────
    print(f"\n[STAGE 3] Parsing key components...")
    t, p, q, s = parse_key_components(combined_digits)
    print(f"  t = {t}   (Collatz seed, 7 digits)")
    print(f"  p = {p}     (page index 1, 3 digits)")
    print(f"  q = {q}     (page index 2, 3 digits)")
    print(f"  s = {s}      (offset, 2 digits)")

    # ── Stage 4: Run 4 Collatz sequences ────────────────────
    print(f"\n[STAGE 4] Running Collatz sequences...")

    # r1 → seq from t,   index p
    seq_t   = collatz_sequence(t)
    print(f"  seq(t={t})     → {len(seq_t)} steps  [used for r1 at index p={p}]")

    # x1 → seq from t+q, index p-s
    seq_t_q = collatz_sequence(t + q)
    print(f"  seq(t+q={t+q})  → {len(seq_t_q)} steps  [used for x1 at index p-s={max(0,p-s)}]")

    # r2 → seq from t,   index q  (reuses seq_t)
    print(f"  seq(t={t})     → reused       [used for r2 at index q={q}]")

    # x2 → seq from t+p, index p+s
    seq_t_p = collatz_sequence(t + p)
    print(f"  seq(t+p={t+p})  → {len(seq_t_p)} steps  [used for x2 at index p+s={p+s}]")

    # ── Stage 5: Extract LSD digits ─────────────────────────
    print(f"\n[STAGE 5] Extracting LSD digits...")

    r1_raw = extract_lsd_digits(seq_t,   start_index=p)
    print(f"  r1_raw  (seq_t   at index p={p})     = {r1_raw:.15f}")

    x1     = extract_lsd_digits(seq_t_q, start_index=max(0, p - s))
    print(f"  x1      (seq_t+q at index p-s={max(0,p-s)})  = {x1:.15f}")

    r2_raw = extract_lsd_digits(seq_t,   start_index=q)
    print(f"  r2_raw  (seq_t   at index q={q})     = {r2_raw:.15f}")

    x2     = extract_lsd_digits(seq_t_p, start_index=p + s)
    print(f"  x2      (seq_t+p at index p+s={p+s})  = {x2:.15f}")

    # ── Stage 6: Map r values to chaotic range ───────────────
    print(f"\n[STAGE 6] Mapping r values to chaotic range [3.57, 4.00]...")
    r1 = map_to_chaotic_range(r1_raw)
    r2 = map_to_chaotic_range(r2_raw)
    print(f"  r1 = 3.57 + {r1_raw:.4f} × 0.43 = {r1:.15f}")
    print(f"  r2 = 3.57 + {r2_raw:.4f} × 0.43 = {r2:.15f}")

    # ── Final Output ─────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  FINAL CHAOTIC PARAMETERS")
    print(f"{'=' * 55}")
    print(f"  r1 = {r1:.15f}")
    print(f"  r2 = {r2:.15f}")
    print(f"  x1 = {x1:.15f}")
    print(f"  x2 = {x2:.15f}")
    print(f"{'=' * 55}")

    # ── Validate ─────────────────────────────────────────────
    assert 3.57 <= r1 <= 4.00, f"r1 out of chaotic range: {r1}"
    assert 3.57 <= r2 <= 4.00, f"r2 out of chaotic range: {r2}"
    assert 0.0  <= x1 <  1.0,  f"x1 out of range [0,1): {x1}"
    assert 0.0  <= x2 <  1.0,  f"x2 out of range [0,1): {x2}"
    print(f"\n  r1 in [3.57, 4.00] ✓")
    print(f"  r2 in [3.57, 4.00] ✓")
    print(f"  x1 in [0, 1)       ✓")
    print(f"  x2 in [0, 1)       ✓")

    return r1, r2, x1, x2


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    image = sys.argv[1] if len(sys.argv) > 1 else "test1.png"
    r1, r2, x1, x2 = generate_params(image)
