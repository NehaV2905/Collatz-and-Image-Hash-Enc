# Collatz-and-Image-Hash-Enc

## Update: Collatz Sequence from Image Hash

### Added
- Collatz sequence generation
- Conversion of hexadecimal image hash to large integer
- Output of hash value, full sequence, and total steps

### Logic
1. Generate image hash using `SIDH_hash`.
2. Convert hexadecimal hash to integer.
3. Apply Collatz rules:
   - If even → n / 2
   - If odd → 3n + 1
4. Repeat until 1 is reached.
5. Count steps and print the sequence.