import hash
def collatz_sequence(n):
    seq = []
    while n != 1:
        seq.append(n)
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3*n + 1
    seq.append(1)
    return seq

def collatz_steps(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3*n + 1
        steps += 1
    return steps

if __name__ == "__main__":
    h = hash.SIDH_hash("test1.png")
    big_number=int(h,16)
    C=collatz_sequence(big_number)
    print(h+"\n")
    print(C)
    print("Steps:", collatz_steps(big_number))

    
    
