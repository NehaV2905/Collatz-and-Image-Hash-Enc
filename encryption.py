"""
encryption.py
=============
Implements Algorithm 2 (Encryption) from the research paper:
"Generating Powerful Encryption Keys for Image Cryptography
 With Chaotic Maps by Incorporating Collatz Conjecture"

Uses r1, r2, x1, x2 from collatz_sequence.py (Way 1 / V1)

Two building blocks:
    1. generate_1d_vector(r2, x2, length)
       → Used for CONFUSION (row and column shuffling)

    2. generate_2d_clmk(r1, x1, rows, cols)
       → Used for DIFFUSION (XOR pixel values)

Encryption pipeline (4 rounds as per paper):
    For each round:
        ├── CONFUSION  : shuffle rows    using 1D vector
        ├── CONFUSION  : shuffle columns using 1D vector
        └── DIFFUSION  : XOR each colour channel with 2D key

Paper reference - Algorithm 2:
    "for k = 1, k++, while k <= 4
        Get 1D Chaotic vector using r2 & x2 → row shuffling
        Shuffle the rows
        Get 1D Chaotic vector using r2 & x2 → column shuffling
        Shuffle the columns
        Generate 2D Chaotic Logistic map key using r1, x1
        Resize key to match colour matrix size
        XOR resized key with colour matrix
     End for
     Combine encrypted colour matrices"
"""

import numpy as np
from PIL import Image
import collatz_sequence as cs


# ═══════════════════════════════════════════════════════════════
# FUNCTION 1  —  generate_1d_vector(r2, x2, length)
# ═══════════════════════════════════════════════════════════════

def generate_1d_vector(r2, x2, length):
    """
    Generate a 1D chaotic vector using the logistic map.
    This vector is used for CONFUSION — shuffling rows and columns.

    The logistic map equation (Equation 2 from paper):
        X(n+1) = r * X(n) * (X(n) - 1)

    Parameters:
        r2     → control parameter    (from collatz_sequence.py)
        x2     → initial condition    (from collatz_sequence.py)
        length → how many values to generate (= number of rows or cols)

    Returns:
        A 1D numpy array of 'length' chaotic float values in (0, 1)

    How it is used for shuffling:
        - Generate vector of length = number of rows
        - argsort the vector → gives a random permutation of row indexes
        - Reorder rows using that permutation
        Same process repeated for columns.

    Example:
        vector  = [0.82, 0.34, 0.91, 0.12]
        argsort = [3, 1, 0, 2]              ← sorted order of indexes
        This means: new row 0 ← old row 3
                    new row 1 ← old row 1
                    new row 2 ← old row 0
                    new row 3 ← old row 2
    """
    vector = []
    x = x2

    for _ in range(length):
        # Logistic map: X(n+1) = r * X(n) * (X(n) - 1)
        x = r2 * x * (x - 1)

        # Keep value in (0, 1) — take absolute value and mod 1
        x = abs(x) % 1.0

        # Avoid exact 0 or 1 which would collapse the map
        if x == 0.0:
            x = 1e-10
        if x >= 1.0:
            x = 1.0 - 1e-10

        vector.append(x)

    return np.array(vector)


# ═══════════════════════════════════════════════════════════════
# FUNCTION 2  —  generate_2d_clmk(r1, x1, rows, cols)
# ═══════════════════════════════════════════════════════════════

def generate_2d_clmk(r1, x1, rows, cols):
    """
    Generate a 2D Chaotic Logistic Map Key (CLMK).
    This matrix is used for DIFFUSION — XORing with pixel values.

    The logistic map equation (Equation 3 from paper):
        CLM(i, j) = X(n+1) = r * X(n) * (X(n) - 1)

    Parameters:
        r1   → control parameter     (from collatz_sequence.py)
        x1   → initial condition     (from collatz_sequence.py)
        rows → number of rows in key matrix
        cols → number of columns in key matrix

    Returns:
        A 2D numpy array of shape (rows, cols)
        Values scaled to uint8 range [0, 255] for XOR with pixels

    How it is used:
        - Generate 2D matrix of same size as image colour channel
        - XOR each pixel value with corresponding matrix value
        - pixel_encrypted = pixel_original XOR key_value
        - This changes the VALUE of every pixel (diffusion)

    Example:
        pixel value : 150  (binary: 10010110)
        key value   : 59   (binary: 00111011)
        XOR result  : 189  (binary: 10111101)  ← encrypted pixel
    """
    key_matrix = np.zeros((rows, cols), dtype=np.float64)
    x = x1

    for i in range(rows):
        for j in range(cols):
            # Logistic map iteration
            x = r1 * x * (x - 1)

            # Keep in (0, 1)
            x = abs(x) % 1.0

            if x == 0.0:
                x = 1e-10
            if x >= 1.0:
                x = 1.0 - 1e-10

            key_matrix[i, j] = x

    # Scale from [0,1) to [0,255] integer for XOR operation
    key_uint8 = (key_matrix * 256).astype(np.uint8)

    return key_uint8


# ═══════════════════════════════════════════════════════════════
# FUNCTION 3  —  confusion(channel, r2, x2)
# ═══════════════════════════════════════════════════════════════

def confusion(channel, r2, x2):
    """
    CONFUSION block — shuffles pixel POSITIONS.
    Implements the permutation step from Algorithm 2.

    Step 1: Generate 1D vector of length = number of rows
            argsort it → row permutation index
            Shuffle rows using that index

    Step 2: Generate 1D vector of length = number of columns
            argsort it → column permutation index
            Shuffle columns using that index

    Parameters:
        channel → 2D numpy array (one colour channel R, G, or B)
        r2      → chaotic control parameter  (from collatz_sequence.py)
        x2      → chaotic initial condition  (from collatz_sequence.py)

    Returns:
        shuffled_channel → 2D numpy array with rows and columns permuted

    Paper quote:
        "A 1D vector generated using Equation (2) shuffles each
         colour channel's rows and columns in the permutation process"
    """
    rows, cols = channel.shape

    # ── Row shuffling ────────────────────────────────────────
    row_vector = generate_1d_vector(r2, x2, rows)

    # argsort gives the indexes that would sort the vector
    # Using these as a permutation index shuffles the rows
    row_permutation = np.argsort(row_vector)
    shuffled = channel[row_permutation, :]

    # ── Column shuffling ─────────────────────────────────────
    # Use a slightly modified x2 so row and column vectors differ
    col_vector = generate_1d_vector(r2, x2 * 0.99 + 0.001, cols)
    col_permutation = np.argsort(col_vector)
    shuffled = shuffled[:, col_permutation]

    return shuffled, row_permutation, col_permutation


# ═══════════════════════════════════════════════════════════════
# FUNCTION 4  —  diffusion(channel, r1, x1)
# ═══════════════════════════════════════════════════════════════

def diffusion(channel, r1, x1):
    """
    DIFFUSION block — changes pixel VALUES using XOR.
    Implements the XOR step from Algorithm 2.

    Step 1: Generate 2D CLMK of same size as colour channel
    Step 2: XOR every pixel with corresponding key value

    Parameters:
        channel → 2D numpy array (one colour channel R, G, or B)
        r1      → chaotic control parameter  (from collatz_sequence.py)
        x1      → chaotic initial condition  (from collatz_sequence.py)

    Returns:
        diffused_channel → 2D numpy array with pixel values changed

    Paper quote:
        "The Diffusion part performs the encryption by XORing bitwise
         each permuted colour channel with the 2D Chaotic logistic key"
    """
    rows, cols = channel.shape

    # Generate 2D key of same size as channel
    key_2d = generate_2d_clmk(r1, x1, rows, cols)

    # XOR pixel values with key
    diffused = np.bitwise_xor(channel, key_2d)

    return diffused


# ═══════════════════════════════════════════════════════════════
# FUNCTION 5  —  encrypt_image(image_path, r1, r2, x1, x2)
# ═══════════════════════════════════════════════════════════════

def encrypt_image(image_path, r1, r2, x1, x2, rounds=4):
    """
    Full encryption — Algorithm 2 from the paper.

    Applies confusion + diffusion for 'rounds' iterations (default 4)
    on each colour channel (R, G, B) separately.

    Paper quote (Algorithm 2):
        "for k = 1, k++, while k <= 4
            [confusion + diffusion on each channel]
         End for
         Combine the encrypted colour matrices"

    Parameters:
        image_path → path to the plain image
        r1, r2     → chaotic control parameters
        x1, x2     → chaotic initial conditions
        rounds     → number of confusion+diffusion rounds (paper uses 4)

    Returns:
        encrypted_image → PIL Image object (encrypted)
        permutations    → dict storing row/col permutations for decryption
    """

    print("=" * 55)
    print("  ENCRYPTION")
    print("  (Algorithm 2 — 4 rounds confusion + diffusion)")
    print("=" * 55)

    # ── Load image and split into R, G, B channels ───────────
    img    = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)

    R = img_np[:, :, 0].copy()
    G = img_np[:, :, 1].copy()
    B = img_np[:, :, 2].copy()

    print(f"\n  Image size   : {img_np.shape[1]} x {img_np.shape[0]}")
    print(f"  Rounds       : {rounds}")
    print(f"  r1={r1:.6f}  r2={r2:.6f}")
    print(f"  x1={x1:.6f}  x2={x2:.6f}")

    # Store permutations for each round (needed for decryption)
    permutations = {"R": [], "G": [], "B": []}

    # ── Apply rounds ──────────────────────────────────────────
    for k in range(1, rounds + 1):
        print(f"\n  [Round {k}]")

        # Process each colour channel independently
        for channel_name, channel in [("R", R), ("G", G), ("B", B)]:

            # CONFUSION — shuffle rows and columns
            shuffled, row_perm, col_perm = confusion(channel, r2, x2)
            print(f"    {channel_name} — confusion  : rows shuffled {row_perm[:5]}...")

            # DIFFUSION — XOR with 2D key
            encrypted_channel = diffusion(shuffled, r1, x1)
            print(f"    {channel_name} — diffusion  : XOR applied")

            # Store permutations for decryption
            permutations[channel_name].append((row_perm, col_perm))

            # Update channel for next round
            if channel_name == "R":
                R = encrypted_channel
            elif channel_name == "G":
                G = encrypted_channel
            else:
                B = encrypted_channel

    # ── Combine encrypted channels back into image ────────────
    encrypted_np    = np.stack([R, G, B], axis=2)
    encrypted_image = Image.fromarray(encrypted_np.astype(np.uint8), "RGB")

    print(f"\n  Encryption complete.")
    print(f"  Original  pixel sample [0,0]: {img_np[0,0,:]}")
    print(f"  Encrypted pixel sample [0,0]: {encrypted_np[0,0,:]}")

    return encrypted_image, permutations


# ═══════════════════════════════════════════════════════════════
# FUNCTION 6  —  decrypt_image(enc_image, r1, r2, x1, x2, perms)
# ═══════════════════════════════════════════════════════════════

def decrypt_image(encrypted_image, r1, r2, x1, x2, permutations, rounds=4):
    """
    Full decryption — Algorithm 3 from the paper (reverse of Algorithm 2).

    Applies diffusion (reverse XOR) + confusion (reverse shuffle)
    in REVERSE round order.

    Paper quote (Algorithm 3):
        "Like the encryption phase, four consecutive rounds of
         reshuffling and diffusion are performed to restore the
         original value of pixels in each colour channel"

    Parameters:
        encrypted_image → PIL Image object (encrypted)
        r1, r2          → same chaotic parameters used for encryption
        x1, x2          → same chaotic parameters used for encryption
        permutations    → dict of row/col permutations from encryption
        rounds          → must match encryption rounds

    Returns:
        decrypted_image → PIL Image object (should match original)
    """

    print("\n" + "=" * 55)
    print("  DECRYPTION")
    print("  (Algorithm 3 — 4 rounds reverse diffusion + confusion)")
    print("=" * 55)

    img_np = np.array(encrypted_image, dtype=np.uint8)

    R = img_np[:, :, 0].copy()
    G = img_np[:, :, 1].copy()
    B = img_np[:, :, 2].copy()

    # ── Apply rounds in REVERSE ───────────────────────────────
    for k in range(rounds, 0, -1):
        print(f"\n  [Round {k} — reverse]")

        for channel_name, channel in [("R", R), ("G", G), ("B", B)]:

            # REVERSE DIFFUSION — XOR again with same key (XOR is its own inverse)
            undiffused = diffusion(channel, r1, x1)
            print(f"    {channel_name} — reverse diffusion : XOR applied")

            # REVERSE CONFUSION — unshuffle rows and columns
            row_perm, col_perm = permutations[channel_name][k - 1]

            # Inverse permutation: reverse the column shuffle
            inv_col_perm = np.argsort(col_perm)
            unshuffled   = undiffused[:, inv_col_perm]

            # Inverse permutation: reverse the row shuffle
            inv_row_perm = np.argsort(row_perm)
            unshuffled   = unshuffled[inv_row_perm, :]
            print(f"    {channel_name} — reverse confusion : rows/cols restored")

            if channel_name == "R":
                R = unshuffled
            elif channel_name == "G":
                G = unshuffled
            else:
                B = unshuffled

    # ── Combine decrypted channels ────────────────────────────
    decrypted_np    = np.stack([R, G, B], axis=2)
    decrypted_image = Image.fromarray(decrypted_np.astype(np.uint8), "RGB")

    print(f"\n  Decryption complete.")
    print(f"  Decrypted pixel sample [0,0]: {decrypted_np[0,0,:]}")

    return decrypted_image


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT — test with a sample image
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test1.png"

    # ── Step 1: Generate r1, r2, x1, x2 from collatz_sequence.py (V1) ───
    print("\nGenerating chaotic parameters from collatz_sequence.py (V1)...")
    r1, r2, x1, x2 = cs.generate_params(image_path)

    # ── Step 2: Encrypt ──────────────────────────────────────────────────
    encrypted_img, perms = encrypt_image(image_path, r1, r2, x1, x2)
    encrypted_img.save("encrypted.png")
    print("\n  Saved → encrypted.png")

    # ── Step 3: Decrypt ──────────────────────────────────────────────────
    decrypted_img = decrypt_image(encrypted_img, r1, r2, x1, x2, perms)
    decrypted_img.save("decrypted.png")
    print("  Saved → decrypted.png")

    # ── Step 4: Verify ───────────────────────────────────────────────────
    original  = np.array(Image.open(image_path).convert("RGB"))
    decrypted = np.array(decrypted_img)

    mse = np.mean((original.astype(int) - decrypted.astype(int)) ** 2)
    print(f"\n{'=' * 55}")
    print(f"  VERIFICATION")
    print(f"{'=' * 55}")
    print(f"  MSE between original and decrypted : {mse:.6f}")
    if mse == 0.0:
        print(f"  Perfect decryption ✓  (MSE = 0)")
    else:
        print(f"  MSE > 0 — decryption has errors (check rounds/params)")
    print(f"{'=' * 55}")
