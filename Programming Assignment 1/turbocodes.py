import json
import random
########################################################################

# Do not modify the lines above

########################################################################

# You can add any import statements from inbuilt libraries here

import itertools
from collections import deque
import functools
import heapq
from math import inf

########################################################################

# Do not install any more external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################

def normalize(array):
    total = sum(array)
    return [x / total for x in array] if total > 0 else array

generator1=0o5
generator2=0o7


def generate_trellis(generator1, generator2, history_length=3):
    # Convert generators into binary masks (reversed for easy zip alignment)
    feedforward = bin(generator1)[2:][::-1].zfill(history_length)
    feedback = bin(generator2)[2:][::-1].zfill(history_length)

    num_states = 2 ** (history_length - 1)
    trellis = {}

    for state in range(num_states):
        # Represent state as deque of bits (length history_length-1)
        state_bits = [int(x) for x in format(state, f"0{history_length-1}b")]
        trellis[state] = {}

        for input_bit in [0, 1]:
            # History = [oldest, ..., newest]
            history = deque(state_bits + [input_bit], maxlen=history_length)

            # Feedback calculation
            feedback_val = sum(int(b) * int(g) for b, g in zip(history, feedback)) % 2

            # Replace last (input) with feedback result
            history[-1] = feedback_val

            # Parity calculation
            parity = sum(int(b) * int(g) for b, g in zip(history, feedforward)) % 2

            # Next state = drop oldest, keep last (history_length-1) bits
            next_state_bits = list(history)[1:]
            next_state = int("".join(str(x) for x in next_state_bits), 2)

            # Output = (systematic = input_bit, parity)
            trellis[state][input_bit] = (next_state, (input_bit, parity))

    # print(trellis)
    return trellis


def hamming_distance(str1, str2):
    return sum(el1 != el2 for el1, el2 in zip(str1, str2))

def viterbi(noisy_output, probability_matrix, history_length=3):
    '''
        Feel free to add more parameters or delete existing ones. For instance, the generated trellis, etc. 
    '''
    trellis = generate_trellis(generator1, generator2, history_length=3)
    num_states = 2 ** (history_length - 1)
    num_voltage = len(probability_matrix[0])
    length = len(noisy_output)//3
    noisy_output = [
        round(val / (num_voltage - 1))
        for idx, val in enumerate(noisy_output)
        if idx % 3 != 2
    ]
    # print("Noisy output length = ", len(noisy_output))
    # if len(noisy_output) < 100:
    #     print("Noisy Output", noisy_output)

    bm = {}
    for l in range(length):
        bm[l] = {}
        for s in range(num_states):
            bm[l][s] = {}
            for bit in [0, 1]:
                next_state, (_, parity) = trellis[s][bit]
                bm[l][s][bit] = hamming_distance(str(noisy_output[2*l])+str(noisy_output[2*l+1]), str(bit)+str(parity))

    pm = {}
    pm[0] = {0: 0, 1: inf, 2: inf, 3: inf}
    predecessor = {l: {} for l in range(length+1)}

    for l in range(0, length):
        pm[l+1] = {s: float("inf") for s in range(num_states)}
        for s in range(num_states):
            if pm[l][s] == float("inf"):
                continue  # unreachable state

            # Transition with input=0
            next0, (_, parity0) = trellis[s][0]
            new_metric0 = pm[l][s] + bm[l][s][0]
            if new_metric0 < pm[l+1][next0]:
                pm[l+1][next0] = new_metric0
                predecessor[l+1][next0] = (s, 0)

            # Transition with input=1
            next1, (_, parity1) = trellis[s][1]
            new_metric1 = pm[l][s] + bm[l][s][1]
            if new_metric1 < pm[l+1][next1]:
                pm[l+1][next1] = new_metric1
                predecessor[l+1][next1] = (s, 1)

    # --- Backtracking ---
    best_path = []
    current_state = min(pm[length], key=pm[length].get)
    for l in range(length, 0, -1):
        prev_state, bit = predecessor[l][current_state]
        best_path.append(bit)
        current_state = prev_state
    best_path.reverse()
    best_path = [str(b) for b in best_path]

    # print("Length of best_path =", len(best_path))
    # if len(best_path) < 100:
    #     print("BEST PATH =", best_path)

    return best_path


def bcjr(noisy_output, probability_matrix, app_probability):
    '''
        Feel free to add more parameters or delete existing ones. For instance, the transmission error probability matrix, any generated trellis, etc. 
        Here, app_probability refers to the a posteriori probabilities of the bits in the original data, which is estimated by other means.
        app_probability can be of shape (2, T), for instance, where T is the length of the original data.        
    '''
    raise NotImplementedError()

    
class Inference:
    def __init__(self, testcase):
        ### NOTE that you will not have access to the original bitstring in the actual evaluation
        self.noisy_output = testcase["noisy_output"]
        self.length = len(self.noisy_output)//3
        self.probability_matrix = testcase["probability_matrix"]
        self.probability_matrix = [normalize(self.probability_matrix), normalize(self.probability_matrix)[::-1]]
        self.permutation = testcase["permutation"]
        
    def get_viterbi_output(self):
        # print("START")
        # trellis = generate_trellis(generator1, generator2, history_length=3)
        # print("END")
        return ''.join(viterbi(self.noisy_output, self.probability_matrix))

    def get_bcjr_output(self):
        return ''.join(random.choice('01') for _ in range(self.length))

    def get_turbocode_output(self):
        return ''.join(random.choice('01') for _ in range(self.length))


if __name__ == "__main__":
    with open("turbocodes_testcases_neeraje.json", "r") as f:
        data = json.load(f)

    testcases = data["testcases"]

    results = []

    for i, testcase in enumerate(testcases, start=1):
        infer = Inference(testcase)

        viterbi_out = infer.get_viterbi_output()
        bcjr_out = infer.get_bcjr_output()
        turbo_out = infer.get_turbocode_output()
        results.append({
            "testcase": i,
            "viterbi_errors": len([j for j in range(len(viterbi_out)) if viterbi_out[j] != testcase["bitstring"][j]]),
            "bcjr_errors": len([j for j in range(len(bcjr_out)) if bcjr_out[j] != testcase["bitstring"][j]]),
            "turbo_errors": len([j for j in range(len(turbo_out)) if turbo_out[j] != testcase["bitstring"][j]]),
        })

    with open("turbocodes_my_results.json", "w") as f:
        json.dump(results, f, indent=4)