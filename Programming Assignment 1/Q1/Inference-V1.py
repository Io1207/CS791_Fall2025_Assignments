import json
import itertools
import heapq
import math

class Inference:
    def __init__(self, data):
        self.factors_count = data['Factors_Count']
        self.states_count = data['State_Count']
        self.num_observations = data['Number of Observations']
        self.observation_sequence = data['Observation Sequence']
        self.transition_potentials = data['Transition Potentials']
        self.transition_potentials = list(self.transition_potentials.values())
        self.transition_potentials_sum = [[sum(self.transition_potentials[i][j:j+self.states_count]) for j in range(0, len(self.transition_potentials[i]), self.states_count)] for i in range(len(self.transition_potentials))]
        self.transition_potentials = [[potential/self.transition_potentials_sum[i][j//self.states_count] for j, potential in enumerate(self.transition_potentials[i])] for i in range(len(self.transition_potentials))]
        self.state_factor_potentials = data['State_Factor_Potentials']
        self.state_factor_potentials_sum = [sum(self.state_factor_potentials[i:i+self.states_count]) for i in range(0, len(self.state_factor_potentials), self.states_count)]
        self.state_factor_potentials = [potential/self.state_factor_potentials_sum[i//self.states_count] for i, potential in enumerate(self.state_factor_potentials)]

    def state_factor_potential(self, assignment, t):
        F, K = self.factors_count, self.states_count
        index = 0
        obs = self.observation_sequence[t]
        index += obs
        multiplier = K
        for f in range(F-1, -1, -1):
            index += assignment[f] * multiplier
            multiplier *= K
        return self.state_factor_potentials[index]

    def forward_backward(self):
        F, K, T = self.factors_count, self.states_count, self.num_observations

        alpha = [{} for _ in range(T)]
        beta = [{} for _ in range(T)]

        for assignment in itertools.product(range(K), repeat=F):
            alpha[0][assignment] = self.state_factor_potential(assignment, 0)
        for t in range(1, T):
            alpha[t] = {}
            for current in itertools.product(range(K), repeat=F):
                total = 0.0
                for prev in itertools.product(range(K), repeat=F):
                    trans_prob = 1.0
                    for f in range(F):
                        idx = prev[f] * K + current[f]
                        trans_prob *= self.transition_potentials[f][idx]
                    total += trans_prob * alpha[t-1][prev]
                alpha[t][current] = total * self.state_factor_potential(current, t)

        for assignment in itertools.product(range(K), repeat=F):
            beta[T-1][assignment] = 1.0
        for t in reversed(range(T-1)):
            beta[t] = {}
            for current in itertools.product(range(K), repeat=F):
                total = 0.0
                for nxt in itertools.product(range(K), repeat=F):
                    trans_prob = 1.0
                    for f in range(F):
                        idx = current[f] * K + nxt[f]
                        trans_prob *= self.transition_potentials[f][idx]
                    total += trans_prob * self.state_factor_potential(nxt, t+1) * beta[t+1][nxt]
                beta[t][current] = total

        self.alpha = alpha
        self.beta = beta

    def get_z_value(self):
        self.forward_backward()
        return sum(self.alpha[self.num_observations-1].values())

    def compute_marginals(self):
        F, K, T = self.factors_count, self.states_count, self.num_observations
        Z = self.get_z_value()
        marginals = [[0.0 for _ in range(K)] for _ in range(F*T)]

        for t in range(T):
            for assignment, a_val in self.alpha[t].items():
                b_val = self.beta[t][assignment]
                prob = a_val * b_val / Z
                for f in range(F):
                    marginals[t*F + f][assignment[f]] += prob
        return marginals

    def compute_top_k(self, k=2):
        F, K, T = self.factors_count, self.states_count, self.num_observations

        joint_states = list(itertools.product(range(K), repeat=F))
        S = len(joint_states)

        log_emit = [[float('-inf')]*S for _ in range(T)]
        for t in range(T):
            for j, js in enumerate(joint_states):
                val = self.state_factor_potential(js, t)
                log_emit[t][j] = math.log(val) if val > 0.0 else float('-inf')

        trans_log = [[float('-inf')]*S for _ in range(S)]
        for i, prev in enumerate(joint_states):
            for j, curr in enumerate(joint_states):
                p = 1.0
                for f in range(F):
                    idx = prev[f]*K + curr[f]
                    p *= self.transition_potentials[f][idx]
                trans_log[i][j] = math.log(p) if p > 0.0 else float('-inf')

        dp = [[[] for _ in range(S)] for _ in range(T)]

        for j in range(S):
            dp[0][j] = [(log_emit[0][j], [j])]

        for t in range(1, T):
            for j in range(S):
                heap = []  
                for i in range(S):
                    if not dp[t-1][i]:
                        continue
                    tl = trans_log[i][j]
                    le = log_emit[t][j]
                    if tl == float('-inf') or le == float('-inf'):
                        continue
                    for prev_score, prev_path in dp[t-1][i]:
                        if prev_score == float('-inf'):
                            continue
                        cand = prev_score + tl + le
                        item = (cand, prev_path + [j])
                        if len(heap) < k:
                            heapq.heappush(heap, item)    
                        else:
                            if cand > heap[0][0]:
                                heapq.heapreplace(heap, item)
                
                dp[t][j] = sorted(heap, key=lambda x: x[0], reverse=True)

        final_heap = []
        for j in range(S):
            for score, path in dp[T-1][j]:
                if len(final_heap) < k:
                    heapq.heappush(final_heap, (score, path))
                else:
                    if score > final_heap[0][0]:
                        heapq.heapreplace(final_heap, (score, path))
        topk = sorted(final_heap, key=lambda x: x[0], reverse=True)

        Z = self.get_z_value()  
        results = []
        for logscore, path in topk:
            flat = []
            for j_idx in path:
                flat.extend(joint_states[j_idx]) 
            joint_prob = math.exp(logscore) if logscore != float('-inf') else 0.0
            post_prob = joint_prob / Z if Z > 0 else 0.0
            results.append({
                "assignment": flat,
                "probability": post_prob
            })
        return results

    def triangulate_and_get_cliques(self):
        pass

    def get_junction_tree(self):
        pass

    def assign_potentials_to_cliques(self):
        pass

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('TestCases.json')
    evaluator.get_output()
    evaluator.write_output('My_Output.json')