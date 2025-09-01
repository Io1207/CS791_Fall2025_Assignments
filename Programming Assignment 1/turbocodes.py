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
from math import inf, log, exp
import math

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
    trellis = generate_trellis(generator1, generator2, history_length=history_length)
    num_states = 2 ** (history_length - 1)
    length = len(noisy_output)//3
    noisy_output = [
        val
        for idx, val in enumerate(noisy_output)
        if idx % 3 != 2
    ]

    bm = {}
    for l in range(length):
        bm[l] = {}
        for s in range(num_states):
            bm[l][s] = {}
            for bit in [0, 1]:
                next_state, (sys_bit, parity) = trellis[s][bit]

                # Transmitted bits are systematic + parity
                transmitted = [sys_bit, parity]

                # Received (noisy) symbols for this time step
                received = noisy_output[2*l : 2*l+2]

                # Compute negative log likelihood
                neg_log_prob = 0.0
                for tx_bit, rx_val in zip(transmitted, received):
                    # probability that observed rx_val comes from tx_bit
                    prob = probability_matrix[tx_bit][rx_val]

                    # guard against log(0)
                    if prob <= 1e-12:
                        prob = 1e-12

                    neg_log_prob += -math.log(prob)

                bm[l][s][bit] = neg_log_prob

    # Initialize path metrics
    pm = {}
    pm[0] = {s: float("inf") for s in range(num_states)}
    pm[0][0] = 0  # start from state 0
    predecessor = {l: {} for l in range(length+1)}

    # Forward pass
    for l in range(0, length):
        pm[l+1] = {s: float("inf") for s in range(num_states)}
        for s in range(num_states):
            if pm[l][s] == float("inf"):
                continue  # unreachable state

            for bit in [0, 1]:
                next_state, _ = trellis[s][bit]
                new_metric = pm[l][s] + bm[l][s][bit]
                if new_metric < pm[l+1][next_state]:
                    pm[l+1][next_state] = new_metric
                    predecessor[l+1][next_state] = (s, bit)

    # Backtracking
    decoded_bits = []
    # pick best state at the final time
    final_state = min(pm[length], key=pm[length].get)
    state = final_state
    for l in range(length, 0, -1):
        state, bit = predecessor[l][state]
        decoded_bits.append(bit)

    decoded_bits.reverse()
    decoded_bits = [str(b) for b in decoded_bits]

    return decoded_bits

def logsumexp(values):
    """Stable log-sum-exp."""
    m = max(values)
    return m + log(sum(exp(v - m) for v in values))

def bcjr(noisy_output, probability_matrix, trellis):
    num_states = len(trellis)
    length = len(noisy_output) // 2  # since each bit produces 2 symbols

    # Branch metric γ in log-domain
    gamma = [{} for _ in range(length)]
    for k in range(length):
        v_sys, v_par = noisy_output[2*k], noisy_output[2*k+1]
        for s in range(num_states):
            gamma[k][s] = {}
            for bit in [0, 1]:
                next_state, (sys_bit, par_bit) = trellis[s][bit]
                p_sys = probability_matrix[sys_bit][v_sys]
                p_par = probability_matrix[par_bit][v_par]
                gamma[k][s][bit] = {
                    "next": next_state,
                    "logprob": log(p_sys + 1e-300) + log(p_par + 1e-300)
                }

    # Forward recursion α
    alpha = [[float("-inf")] * num_states for _ in range(length + 1)]
    alpha[0][0] = 0.0  # start in state 0
    for k in range(length):
        for s in range(num_states):
            if alpha[k][s] == -inf:  # unreachable
                continue
            for bit, trans in gamma[k][s].items():
                ns = trans["next"]
                val = alpha[k][s] + trans["logprob"]
                alpha[k+1][ns] = logsumexp([alpha[k+1][ns], val])

    # Backward recursion β
    beta = [[float("-inf")] * num_states for _ in range(length + 1)]
    # beta[length][:] = 0.0  # assume all end states equally likely
    beta[length] = [0.0] * num_states
    for k in reversed(range(length)):
        for s in range(num_states):
            vals = []
            for bit, trans in gamma[k][s].items():
                ns = trans["next"]
                vals.append(trans["logprob"] + beta[k+1][ns])
            if vals:
                beta[k][s] = logsumexp(vals)

    # Compute LLRs
    LLRs = []
    for k in range(length):
        num_vals, den_vals = [], []
        for s in range(num_states):
            for bit, trans in gamma[k][s].items():
                ns = trans["next"]
                term = alpha[k][s] + trans["logprob"] + beta[k+1][ns]
                if bit == 1:
                    num_vals.append(term)
                else:
                    den_vals.append(term)
        llr = logsumexp(num_vals) - logsumexp(den_vals)
        LLRs.append(llr)

    # if length < 100:
    #     print(LLRs)

    decoded = [1 if llr > 0 else 0 for llr in LLRs]
    decoded = [str(b) for b in decoded]
    return decoded


def bcjr_helper(noisy_output, probability_matrix, other_llr, trellis, eps=1e-10):
    """
    Returns (extrinsic_llrs, a_posteriori_llrs).
    - noisy_output: list of length 2*T: [v_sys0, v_par0, v_sys1, v_par1, ...]
    - probability_matrix[bit][v] = P(voltage=v | bit)
    - other_llr: list of length T with prior LLRs for the information bits (can be zeros)
    - trellis: mapping trellis[state][bit] -> (next_state, (sys_bit, par_bit))
    """
    NEG_INF = float("-inf")
    num_states = len(trellis)
    length = len(noisy_output) // 2
    # print("Length = ", length)
    # print("other_llr = ", other_llr)
    if other_llr is None:
        other_llr = [0.0] * length
    assert len(other_llr) == length

    # Build gamma (branch metrics) including prior term
    gamma = [ {s: {} for s in range(num_states)} for _ in range(length) ]
    for k in range(length):
        v_sys = noisy_output[2*k]
        v_par = noisy_output[2*k + 1]
        L_prior = other_llr[k]
        for s in range(num_states):
            for bit in (0, 1):
                ns, (sys_bit, par_bit) = trellis[s][bit]
                p_sys = probability_matrix[sys_bit][v_sys]
                p_par = probability_matrix[par_bit][v_par]
                # safe logs for channel likelihood
                lp = math.log(max(p_sys, eps)) + math.log(max(p_par, eps))
                
                # log P(u_k = bit) from other_llr (stable)
                if bit == 1:
                    log_prior = -math.log1p(math.exp(-L_prior))  # ln(sigmoid(L))
                else:
                    log_prior = -math.log1p(math.exp(L_prior))   # ln(1 - sigmoid(L))

                gamma[k][s][bit] = {"next": ns, "logprob": lp + log_prior}

    # Forward recursion (alpha)
    alpha = [[NEG_INF] * num_states for _ in range(length + 1)]
    alpha[0][0] = 0.0
    for k in range(length):
        for s in range(num_states):
            if alpha[k][s] == NEG_INF:
                continue
            for bit, trans in gamma[k][s].items():
                ns = trans["next"]
                val = alpha[k][s] + trans["logprob"]
                alpha[k + 1][ns] = logsumexp([alpha[k + 1][ns], val])

    # Backward recursion (beta)
    beta = [[NEG_INF] * num_states for _ in range(length + 1)]
    beta[length] = [0.0] * num_states
    for k in range(length - 1, -1, -1):
        for s in range(num_states):
            vals = []
            for bit, trans in gamma[k][s].items():
                ns = trans["next"]
                vals.append(trans["logprob"] + beta[k + 1][ns])
            beta[k][s] = logsumexp(vals) if vals else NEG_INF

    # a-posteriori LLRs (include prior because gamma included it)
    a_posteriori = []
    for k in range(length):
        num_vals = []
        den_vals = []
        for s in range(num_states):
            for bit, trans in gamma[k][s].items():
                ns = trans["next"]
                term = alpha[k][s] + trans["logprob"] + beta[k + 1][ns]
                if bit == 1:
                    num_vals.append(term)
                else:
                    den_vals.append(term)
        L_app = logsumexp(num_vals) - logsumexp(den_vals)
        a_posteriori.append(L_app)

    # compute systematic-only channel LLRs: ln(P(v_sys|1)/P(v_sys|0))
    sys_llr = []
    for k in range(length):
        v_sys = noisy_output[2*k]
        p1 = probability_matrix[1][v_sys]
        p0 = probability_matrix[0][v_sys]
        sys_llr.append(math.log(max(p1, eps)) - math.log(max(p0, eps)))

    # extrinsic = L_aposteriori - prior - sys_llr
    extrinsic = [L_app - L_prior - s_llr for L_app, L_prior, s_llr
                 in zip(a_posteriori, other_llr, sys_llr)]

    decoded = [1 if llr > 0 else 0 for llr in a_posteriori]
    decoded = [str(b) for b in decoded]
    
    return extrinsic, decoded

def unshuffle(shuffled, perm):
    n = len(perm)
    original = [None] * n
    for i, p in enumerate(perm):
        original[p] = shuffled[i]
    return original

def interleave(array1, array2, array3=None):
    interleaved = []
    for i in range(len(array1)):
        interleaved.append(array1[i])
        interleaved.append(array2[i])
        if array3 is not None:
            interleaved.append(array3[i])
    return interleaved
    
    
class Inference:
    def __init__(self, testcase):
        ### NOTE that you will not have access to the original bitstring in the actual evaluation
        self.noisy_output = testcase["noisy_output"]
        self.length = len(self.noisy_output)//3
        self.probability_matrix = testcase["probability_matrix"]
        self.probability_matrix = [normalize(self.probability_matrix), normalize(self.probability_matrix)[::-1]]
        self.permutation = testcase["permutation"]
        
    def get_viterbi_output(self):
        return ''.join(viterbi(self.noisy_output, self.probability_matrix))

    def get_bcjr_output(self):
        one_parity_noisy_output = [val for idx, val in enumerate(self.noisy_output) if idx % 3 != 2]
        return ''.join(bcjr(one_parity_noisy_output, self.probability_matrix, generate_trellis(generator1, generator2, history_length=3)))

    def get_turbocode_output(self):
        first_channel = [val for idx, val in enumerate(self.noisy_output) if idx % 3 != 2]
        systematic = [val for idx, val in enumerate(self.noisy_output) if idx % 3 == 0]
        shuffled_systematic = [systematic[self.permutation[i]] for i in range(len(self.permutation))]
        parity2 = [val for idx, val in enumerate(self.noisy_output) if idx % 3 == 2]
        # unshuffled_parity2 = unshuffle(parity2, self.permutation)
        second_channel = interleave(shuffled_systematic, parity2)
        trellis = generate_trellis(generator1, generator2, history_length=3)

        llr1 = [0.0] * self.length  # prior for decoder 2 (initially zero)
        llr2 = [0.0] * self.length
        ap1 = None

        epochs = 7
        for epoch in range(epochs):
            llr2, decoded2 = bcjr_helper(second_channel, self.probability_matrix, llr1, trellis)
            llr2 = unshuffle(llr2, self.permutation)
            llr1, decoded1 = bcjr_helper(first_channel,  self.probability_matrix, llr2, trellis)
            llr1 = [llr1[self.permutation[i]] for i in range(len(self.permutation))]
            # now llr1 is extrinsic from decoder1 (to be used next iteration by decoder2)
        
        return ''.join(decoded1)
    


if __name__ == "__main__":
    with open("turbocodes_testcases.json", "r") as f:
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

    with open("turbocodes_results.json", "w") as f:
        json.dump(results, f, indent=4)
