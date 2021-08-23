from argparse import ArgumentParser
import numpy as np
from pathlib import Path


# A function that takes a matrix (H) and add ones in one of its rows (row) at specific places (list)
def add_ones(H, row, list):
    for item in list:
        H[row][item] = 1
    return H

if __name__ == "__main__":
    
    parser = ArgumentParser(description="Create surface codes.")
    parser.add_argument("--length", "-l", type=int, help="Length of the surface code.")
    
    arguments = parser.parse_args()
    n = arguments.length

    # n minus 1
    nm1 = n-1

    # number of x/z stabilisers
    stabilisers = n * nm1

    qubits = n*n + nm1 * nm1

    Hx = np.zeros((stabilisers,qubits))
    Hz = np.zeros((stabilisers,qubits))

    # Creation of the Hz and Hz matrices after finding the patterns in the lattice of the surface code.

    # Create the Hz matrix 
    for i in range(stabilisers):
        ones = []
        mod = i % n
        div = i // n

        # Find the edges (qubits) that surround a face (Z-stabiliser)
        ones.append(div * (n + nm1) + mod)
        ones.append(div * (n + nm1) + n + mod - 1)
        ones.append(div * (n + nm1) + n + mod)
        ones.append(div * (n + nm1) + n + nm1 + mod)

        # If the Z-stabiliser is on the edge (3 qubits instead of 4)
        # remove the appropriate edge
        if i % n == 0:
            del ones[1]
        if (i + 1) % n == 0:
            del ones[2]
        
        add_ones(Hz, i, ones)

    # Create the Hx matrix 
    for i in range(stabilisers):
        ones = []
        mod = i % nm1
        div = i // nm1

        # Find the edges (qubits) that are incident to every vertex (X-stabiliser)
        ones.append((div - 1) * nm1 + div * n + mod)
        ones.append(div * nm1 + div * n + mod)
        ones.append(div * nm1 + div * n + mod + 1)
        ones.append(div * nm1 + (div + 1) * n + mod)

        # Take care of the case where an X-stabiliser has 3 incident qubits
        ones = [i for i in ones if i >= 0 and i < qubits]
        add_ones(Hx, i, ones)

    # Create the logical operators
    logx = np.zeros((1,qubits))
    logz = np.zeros((1,qubits))

    middle = n // 2

    ones = []
    for i in range(n):
        ones.append(i + (n*middle + nm1 * middle))
    add_ones(logz,0,ones)


    ones = []
    ones.append(middle)
    for i in range(n-1):
        last_elem = ones[-1]
        ones.append(last_elem + n + nm1)
    add_ones(logx,0,ones)

    # Save all the info to the appropriate files
    path = 'Hyperbolic/{4,4}/4/Stabilisers/x'
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = path + f'/{qubits}.csv'
    np.savetxt(filename, Hx, delimiter=',', fmt='%0.0f')
    
    path = 'Hyperbolic/{4,4}/4/Stabilisers/z'
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = path + f'/{qubits}.csv'
    np.savetxt(filename, Hz, delimiter=',', fmt='%0.0f')
    
    path = 'Hyperbolic/{4,4}/4/Logicals/x'
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = path + f'/{qubits}.csv'
    np.savetxt(filename, logx, delimiter=',', fmt='%0.0f')
    
    path = 'Hyperbolic/{4,4}/4/Logicals/z'
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = path + f'/{qubits}.csv'
    np.savetxt(filename, logz, delimiter=',', fmt='%0.0f')

    distance = np.array([n])
    path =     path = 'Hyperbolic/{4,4}/4/Logicals'
    filename = path + '/x/' + f'd{qubits}.csv'
    np.savetxt(filename, distance, delimiter=',', fmt='%0.0f')
    filename = path + '/z/' + f'd{qubits}.csv'
    np.savetxt(filename, distance, delimiter=',', fmt='%0.0f')

