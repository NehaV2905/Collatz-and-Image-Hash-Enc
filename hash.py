import numpy as np
from PIL import Image
from scipy.fftpack import dct
import hashlib

def entropy(block):
    hist = np.histogram(block, bins=256, range=(0,1))[0]
    p = hist / np.sum(hist)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def logistic_sine(x, r):
    return np.sin(np.pi * r * x * (1 - x))


def SIDH_hash(image_path, salt=b""):
    #Preprocess
    img = Image.open(image_path).convert("L").resize((256,256))
    I = np.array(img) / 255.0

    #Block statistics
    blocks = []
    size = 16
    for i in range(0,256,size):
        for j in range(0,256,size):
            block = I[i:i+size, j:j+size]
            blocks.extend([
                entropy(block),
                np.mean(block),
                np.var(block)
            ])

    B = np.array(blocks)

    #Spectral signature
    F = dct2(I)

    low = F[:8,:8].flatten()
    high = F[-8:,-8:].flatten()

    S = np.concatenate([low, high])

    #Seed hash
    seed_data = B.tobytes() + S.tobytes() + salt
    X = hashlib.sha3_256(seed_data).digest()
    X_bytes = np.frombuffer(X, dtype=np.uint8)

    #Chaotic generator
    x0 = int.from_bytes(X[:8],'big') / 2**64
    r  = 3.9 + (int.from_bytes(X[8:16],'big') / 2**64) * 0.099999
    C = []
    x = x0
    for _ in range(len(X_bytes)):
        x = logistic_sine(x, r)
        C.append(x)

    C = np.array(C)

    #Avalanche diffusion

    Y = X_bytes.astype(int)

    for _ in range(4):
        for i in range(len(Y)):
            c = int(C[i]*256) & 0xFF

            Y[i] ^= c

            shift = c % 8
            Y[i] = ((Y[i] << shift) | (Y[i] >> (8-shift))) & 0xFF

            Y[i] = (Y[i] + int(C[(i+1)%len(C)]*255)) & 0xFF
    #Bit permutation
    Y_uint8 = np.array(Y, dtype=np.uint8)
    bits = np.unpackbits(Y_uint8)
    perm = np.argsort(C)
    bits = bits[perm]

    #Final compression
    final_bytes = np.packbits(bits).tobytes()
    return hashlib.blake2s(final_bytes).hexdigest()


if __name__ == "__main__":
    h = SIDH_hash("test.png")
    print("Hash:", h)
