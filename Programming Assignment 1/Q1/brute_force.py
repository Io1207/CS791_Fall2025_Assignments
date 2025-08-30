import json
import itertools
import heapq

########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################




class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        
        Refer to the sample test case for the structure of the input data.
        """
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

    def triangulate_and_get_cliques(self):
        """
        Triangulate the undirected graph and extract the maximal cliques.
        
        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.

        Refer to the problem statement for details on triangulation and clique extraction.
        """
        pass

    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """
        pass

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.
        """
        pass

    def assignment_prob(self, assignment):
        prob = 1.0
        F = self.factors_count
        K = self.states_count
        T = self.num_observations

        for f in range(F):           
            for t in range(T - 1):     
                s_prev = assignment[t * F + f]
                s_next = assignment[(t + 1) * F + f]
                idx = s_prev * K + s_next  
                prob *= self.transition_potentials[f][idx]

        for t in range(T):
            index = 0
            
            obs = self.observation_sequence[t]
            index += obs

            multiplier = K

            # Hidden factor states
            for f in range(F - 1, -1, -1):
                s = assignment[t * F + f]
                index += s * multiplier
                multiplier *= K

            prob *= self.state_factor_potentials[index]

        return prob

    def get_z_value(self):
        F, K, T = self.factors_count, self.states_count, self.num_observations
        total = 0.0
        for assignment in itertools.product(range(K), repeat=F*T):
            total += self.assignment_prob(assignment)
        return total

    def compute_marginals(self):
        F, K, T = self.factors_count, self.states_count, self.num_observations
        Z = self.get_z_value()
        
        marginals = [[0.0 for _ in range(K)] for _ in range(F*T)]
        
        for assignment in itertools.product(range(K), repeat=F*T):
            p = self.assignment_prob(assignment)
            for idx, s in enumerate(assignment):
                marginals[idx][s] += p
        for idx in range(F*T):
            for s in range(K):
                marginals[idx][s] /= Z
        return marginals


    def compute_top_k(self, k=3):
        F, K, T = self.factors_count, self.states_count, self.num_observations
        Z = self.get_z_value()
        heap = []
        
        for assignment in itertools.product(range(K), repeat=F*T):
            p = self.assignment_prob(assignment) / Z
            heapq.heappush(heap, (-p, list(assignment)))
        
        result = []
        for _ in range(min(k, len(heap))):
            p, assignment = heapq.heappop(heap)
            result.append({"assignment": assignment, "probability": -p})
        
        return result



########################################################################

# Do not change anything below this line

########################################################################

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