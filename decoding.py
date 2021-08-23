import numpy as np
import scipy
from pymatching import Matching
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from argparse import ArgumentParser
from shapely.geometry import LineString
import re
from time import process_time
import psutil
import json

def ram_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_percent()
    return mem

# noiseless parity check measurements
def num_decoding_failures(H, logicals, p, num_trials):
    """ 
    H --> is the matrix that describes the lattice (Hz or Hx)
    logicals --> X-logical or Z-logical operators
    p --> list of the error thresholds to test
    num_trials --> the number or repetitions of the decoder
    """

    # It's a good practice to create the Matchin object at the beginning of the procedure.
    matching = Matching(H)
    num_errors = 0

    # For every point defined by linspace we run the decoding procedure num_trials times.
    for i in range(num_trials):

        # Simulate an error under the independent noise model
        # where an error will occur with probability p. 
        noise = np.random.binomial(1, p, H.shape[1])

        # The syndrome of the stabilisers is calculated 
        # from the dot product (modulo 2) between noise and H.
        syndrome = H@noise % 2

        # Use PyMatching to run the decoder (MWPM algorithm).
        correction = matching.decode(syndrome)
        
        # The error is the sum (modulo 2) between noise and correction.
        error = (noise + correction) % 2

        if np.any(error@logicals.T % 2):
            num_errors += 1

    return num_errors

# noisy parity check measurements
def num_decoding_failures_noisy_syndromes(H, logicals, p, q, num_trials, repetitions):
    """
    H -->  matrix that describes the lattice (Hz or Hx)
    logicals --> X-logical or Z-logical operators matrix
    p --> list of the error thresholds to test
    num_trials --> the number or repetitions of the decoder
    p --> The probability that a qubit has error
    q --> The probability that the error syndrome is measured incorrectly
    repetitions --> The number of repetitions T (or the number that the stabilisers measurements are repeated)
    It can be seen as the number of vertical levels during the decoding
    """
    # spacelike_weights refers to the weight of edges in the matching graph (default is None which means weight = 1.0). 
    #  timelike_weights gives the weight of timelike edges.
    matching = Matching(H, spacelike_weights=np.log((1-p)/p),
                repetitions=repetitions, timelike_weights=np.log((1-q)/q))

    num_stabilisers, num_qubits = H.shape

    
    num_errors = 0
    
    for i in range(num_trials):

        # qubit error
        noise_new = (np.random.rand(num_qubits, repetitions) < p).astype(np.uint8)
        noise_cumulative = (np.cumsum(noise_new, 1) % 2).astype(np.uint8)
        noise_total = noise_cumulative[:,-1]
        
        # syndrome error
        syndrome = H@noise_cumulative % 2
        syndrome_error = (np.random.rand(num_stabilisers, repetitions) < q).astype(np.uint8)
        
        # Last round has perfect measurement to guarantee even parity.
        syndrome_error[:,-1] = 0 

        noisy_syndrome = (syndrome + syndrome_error) % 2

        # Convert to difference syndrome (the latest).
        noisy_syndrome[:,1:] = (noisy_syndrome[:,1:] - noisy_syndrome[:,0:-1]) % 2
        
        correction = matching.decode(noisy_syndrome)
        
        error = (noise_total + correction) % 2
        
        assert not np.any(H@error % 2)
        
        if np.any(error@logicals.T % 2):
            num_errors += 1

    return num_errors


# A function that first run the decoding and then creates the appropriate plots.
def decode_and_plot(rsnt,ps,xz,noise):
    
    """
    rsnt is a list that containts r, s, num_cuts, num_trials
    
    rsnt[0] --> r
    rsnt[1] --> s
    rsnt[2] --> num_cuts
    rsnt[3] --> num_trial
    ps --> is a list with all the physical error rates to test
    xz --> is x or z depending on the type of error (Pauli-X or Pauli-Z)
    noise --> 0 or 1 for noiseless and noisy parity check measurements respectively
    """

    np.random.seed(2)

    path = f"Hyperbolic/{{{rsnt[0]},{rsnt[1]}}}/{rsnt[2]}/Stabilisers" 
    path_log = f"Hyperbolic/{{{rsnt[0]},{rsnt[1]}}}/{rsnt[2]}/Logicals"

    # Take the all the csv files and sort them by size. 
    # It is reasonable to assume that file '18.csv' will be less in size than file '76.csv'
    # In that way we sort the files by accenting qubits.
    Lxz = sorted(glob.glob(path + "/" + xz + "/" + "*.csv"), key=os.path.getsize)

    # The elements in Lxz have the form 'Hyperbolic/{6,4}/8/Stabilisers/x/16.csv'
    # So we keep just the last part (16.csv) by splitting the string in / 
    Lxz = [i.split("/")[-1] for i in Lxz]

    # This list keeps all logical errors from different code sizes (e.g. 16.csv, 72.csv, 85.csv ...)
    all_log_errors = []
    
    # Set the timer before decoding
    start = process_time()

    # Measure CPU percentage before decoding
    psutil.cpu_percent()
    
    for L in Lxz:

        print(f"Simulating f={L}...")
        
        # Take the H matrix from file.
        file_name = path + "/" + xz + "/" + L
        H = np.genfromtxt(file_name, delimiter=',')
        H = scipy.sparse.csr_matrix(H)

        # Take the logical operators from file.
        file_name = path_log + "/" + xz + "/" + L
        log = np.genfromtxt(file_name, delimiter=',')
        log = scipy.sparse.csr_matrix(log)  

        # A list that holds the number of errors for every probability p.
        log_errors = []

        # Take the code's distance
        file_name = path_log + "/" + xz + "/" + "d" + L
        distance = np.genfromtxt(file_name) 

        # Run the decoding for every point in the linspace.
        for p in ps:
            if noise:
                num_errors = num_decoding_failures_noisy_syndromes(H, log, p, p, rsnt[3], int(distance))
                log_errors.append(num_errors/rsnt[3])
            else:
                num_errors = num_decoding_failures(H, log, p, rsnt[3])
                log_errors.append(num_errors/rsnt[3])

        all_log_errors.append(np.array(log_errors))
                
    # Measure time, CPU and RAM after the decoding
    stop = process_time()
    CPU = psutil.cpu_percent()
    RAM = ram_usage()

    metadata = {}
    metadata['CPU'] = CPU
    metadata['RAM'] = RAM
    metadata['TIME'] = stop - start

    # Find the intersection points. 

    # This is a line that we use as a reference. It will be the line that comes up after the decoding and has the highest number of qubits.
    # We will search for intersection points between this line and every other line. 
    reference_line = LineString(np.column_stack((ps, all_log_errors[-1]))) 

    all_intersection_points = []
    all_lines = all_log_errors.copy()[0:-1]

    for line_points in all_lines:
        line = LineString(np.column_stack((ps, line_points)))
        intersection = reference_line.intersection(line)
        #all_intersection_points.append(re.sub("[^\d\.\,\s]", "", intersection.wkt))
        all_intersection_points.append(intersection)

    plt.figure()

    for L, logical_errors in zip(Lxz, all_log_errors):    
        
        # Split the file name relative to dot 
        # e.g. 55.csv will be ['55', 'csv']
        parts = L.split(".")

        plt.plot(ps, logical_errors, 'o-', label=f"Q={parts[0]}")

    if xz == 'x':
        xz_opp = 'z' # xz_opp means opposite of xz.
    elif xz == 'z':
        xz_opp = 'x'

    if noise:
        plt_title = f"Simulation of {{{rsnt[0]},{rsnt[1]}}} for {rsnt[2]} boundary cuts\ncorrecting for {xz_opp} logical errors (noisy)."
        noise_type = 'noisy'
    else:
        plt_title = f"Simulation of {{{rsnt[0]},{rsnt[1]}}} for {rsnt[2]} boundary cuts\ncorrecting for {xz_opp} logical errors (noiseless)."
        noise_type = 'noiseless'

    plt.title(plt_title)
    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    plt.legend(loc=0)
    plt.tight_layout()
    
    # Save the figure.
    path_fig = f"Figures/{{{rsnt[0]},{rsnt[1]}}}/{rsnt[2]}" 
    filepath = path_fig + "/correct_" + xz_opp + f"_errors_{noise_type}.png"
    Path(path_fig).mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)

    # Save the intersection points
    filepath = path_fig + "/dec_data_" + xz_opp + f"_errors_{noise_type}.csv"
    np.savetxt(filepath, np.array(all_log_errors))
    filepath = path_fig + "/intersection_points_" + xz_opp + f"_{noise_type}.txt"

    with open(filepath, 'w') as f:
        for item in all_intersection_points:
            f.write("%s\n" % item)

    # Save the time, CPU and RAM data
    filepath = path_fig + "/metadata_" + xz_opp + f"_{noise_type}.txt" 
    with open(filepath, 'w') as file:
        file.write(json.dumps(metadata))

if __name__ == "__main__":
    
    parser = ArgumentParser(description="Run the decoder for hyperbolic codes.")
    parser.add_argument("--r", "-r", type=int, help="Number of r-gons.")
    parser.add_argument("--s","-s", type=int, help="Number of r-gons that meet at a vertex.")
    parser.add_argument("--num_cuts","-n", type=int, help="Number of cuts on the boundary.")
    parser.add_argument("--trials", "-t", type=int, help="Number of times to run the decoder for the same error probability.")
    parser.add_argument("--low","-l", type=float, help="Probability's lower limit.")
    parser.add_argument("--up","-u", type=float, help="Probability's higher limit.")
    parser.add_argument("--pieces","-p", type=int, help="Cut the interval between upper and lower probability limit into p pieces.")
    parser.add_argument("--noise","-e", nargs='?', type=int, default=0, help="Noisy syndrome measurements (0) or not (1).")
    arguments = parser.parse_args()

    r = arguments.r
    s = arguments.s
    num_cuts = arguments.num_cuts
    num_trials = arguments.trials
    noise = arguments.noise
    ps = np.linspace(arguments.low, arguments.up, arguments.pieces)


    """ r = 6
    s = 6
    num_cuts = 4
    num_trials = 1000
    noise = 0
    ps = np.linspace(0.16, 0.2, 10) """

    rsnt = [r,s,num_cuts,num_trials]

    decode_and_plot(rsnt,ps,'x',noise)
    decode_and_plot(rsnt,ps,'z',noise)