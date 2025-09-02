import json
import itertools
import heapq
import math

class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        """
        self.factors_count = data['Factors_Count']
        self.states_count = data['State_Count']
        self.num_observations = data['Number of Observations']
        self.observation_sequence = data['Observation Sequence']
        
        # Process transition potentials - each chain has K×K values
        self.transition_potentials = data['Transition Potentials']
        self.transition_potentials = list(self.transition_potentials.values())
        
        # Normalize each row of each transition matrix
        self.normalized_transition_potentials = []
        for chain_idx in range(self.factors_count):
            chain_potentials = self.transition_potentials[chain_idx]
            normalized_chain = []
            for i in range(self.states_count):
                row_start = i * self.states_count
                row_end = row_start + self.states_count
                row = chain_potentials[row_start:row_end]
                row_sum = sum(row)
                normalized_row = [p/row_sum for p in row] if row_sum > 0 else row
                normalized_chain.extend(normalized_row)
            self.normalized_transition_potentials.append(normalized_chain)
        
        # Process state factor potentials - should have K^(M+1) values
        self.state_factor_potentials = data['State_Factor_Potentials']
        
        # Normalize by grouping into sets of K values (for each observation value)
        self.normalized_state_potentials = []
        num_groups = len(self.state_factor_potentials) // self.states_count
        for i in range(num_groups):
            group_start = i * self.states_count
            group_end = group_start + self.states_count
            group = self.state_factor_potentials[group_start:group_end]
            group_sum = sum(group)
            normalized_group = [p/group_sum for p in group] if group_sum > 0 else group
            self.normalized_state_potentials.extend(normalized_group)
        
        # Create cliques and junction tree
        self.cliques = []
        self.connections = []
        self.create_junction_tree()
        self.assign_potentials_to_cliques()
        
        # Initialize messages
        self.messages = {}
        self.messages_forward = {}  # Forward messages
        self.messages_backward = {}  # Backward messages

    def create_junction_tree(self):
        """
        Create the junction tree structure for the FHMM.
        Only C_1 to C_T observation cliques and separation cliques between them.
        """
        M = self.factors_count
        T = self.num_observations
        
        # Create the junction tree in the correct order: C_1 -> S_1^1 -> S_1^2 -> ... -> S_1^M -> C_2 -> S_2^1 -> ... -> C_T
        for t in range(1, T + 1):
            # Add observation clique C_t
            variables_Ct = [f'H_{t}^{m+1}' for m in range(M)] + [f'O_{t}']
            self.cliques.append({'name': f'C_{t}', 'variables': variables_Ct, 'type': 'observation'})
            
            # If this is not the last time step, add separation cliques
            if t < T:
                for n in range(M):
                    variables_Stn = []
                    # Add H_t^{n+1} to H_t^M
                    for m in range(n, M):
                        variables_Stn.append(f'H_{t}^{m+1}')
                    # Add H_{t+1}^1 to H_{t+1}^{n+1}
                    for m in range(n + 1):
                        variables_Stn.append(f'H_{t+1}^{m+1}')
                    
                    self.cliques.append({'name': f'S_{t}^{n+1}', 'variables': variables_Stn, 'type': 'separation'})
        
        # Create connections (linear chain) in the order they were added
        for i in range(len(self.cliques) - 1):
            self.connections.append((self.cliques[i]['name'], self.cliques[i + 1]['name']))
            # print(self.cliques[i]['name'], self.cliques[i + 1]['name'])

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        The initial state distribution P(H_1^1, H_1^2, ..., H_1^M) is incorporated into C_1.
        """
        M = self.factors_count
        T = self.num_observations
        obs_sequence = self.observation_sequence
        
        for clique in self.cliques:
            name = clique['name']
            clique['potential_func'] = self.create_potential_function(name, obs_sequence, M)

    def create_potential_function(self, clique_name, obs_sequence, M):
        """
        Create a potential function for the given clique.
        """
        K = self.states_count
        
        if clique_name.startswith('C_'):
            # Observation clique - format: C_t
            t = int(clique_name.split('_')[1])
            o_val = obs_sequence[t - 1]  # observations are 0-indexed, cliques are 1-indexed
            
            def observation_potential(assignments):
                # Get hidden state values
                h_vals = [assignments.get(f'H_{t}^{m+1}', 0) for m in range(M)]
                
                # Calculate the index in the state factor potentials
                # The index is: h1 * K^(M-1) + h2 * K^(M-2) + ... + hM * K^0 + o * K^M
                index = o_val
                for m, h_val in enumerate(h_vals):
                    index += h_val * (K ** (M - m))

                return self.normalized_state_potentials[index]
            
            return observation_potential
        
        elif clique_name.startswith('S_'):
            # Separation clique - format: S_t^n
            parts = clique_name.split('_')
            t = int(parts[1].split('^')[0])  # Extract t from "t^n"
            n = int(parts[1].split('^')[1]) - 1  # Convert to 0-indexed
            
            def transition_potential(assignments):
                h_t_n = assignments.get(f'H_{t}^{n+1}', 0)
                h_t1_n = assignments.get(f'H_{t+1}^{n+1}', 0)
                
                # Get transition probability for chain n
                return self.normalized_transition_potentials[n][h_t_n * K + h_t1_n]
            
            return transition_potential
        
        # Default potential function
        return lambda assignments: 1.0

    def get_clique_by_name(self, name):
        """Get clique by its name."""
        for clique in self.cliques:
            if clique['name'] == name:
                return clique
        return None

    def get_sepset_variables(self, clique1, clique2):
        """Get the variables in the separation set between two cliques."""
        return [var for var in clique1['variables'] if var in clique2['variables']]

    def compute_message(self, from_clique, to_clique):
        """Compute message from one clique to another."""
        # Get separation set variables
        sepset_vars = self.get_sepset_variables(from_clique, to_clique)
        
        # Create all possible assignments for the separation set
        assignments_list = []
        for var in sepset_vars:
            if var.startswith('H_'):
                assignments_list.append(range(self.states_count))
            else:  # Observation variable
                t = int(var.split('_')[1])
                assignments_list.append([self.observation_sequence[t - 1]])  # observations are 0-indexed
        
        # Generate all combinations
        all_assignments = list(itertools.product(*assignments_list))
        
        # Initialize message as a dictionary mapping sepset assignments to values
        message = {}
        
        for assignment in all_assignments:
            # Create assignment dictionary
            assignment_dict = dict(zip(sepset_vars, assignment))
            
            # Sum over all variables not in sepset
            message_value = 0.0
            
            # Get variables to sum over (variables in from_clique but not in sepset)
            sum_vars = [var for var in from_clique['variables'] if var not in sepset_vars]
            
            if not sum_vars:
                # No variables to sum over, just compute the potential
                message_value = from_clique['potential_func'](assignment_dict)
            else:
                # Create all possible assignments for variables to sum over
                sum_assignments_list = []
                for var in sum_vars:
                    if var.startswith('H_'):
                        sum_assignments_list.append(range(self.states_count))
                    else:  # Observation variable
                        t = int(var.split('_')[1])
                        sum_assignments_list.append([self.observation_sequence[t - 1]])  # observations are 0-indexed
                
                # Generate all combinations for sum variables
                all_sum_assignments = list(itertools.product(*sum_assignments_list))
                
                for sum_assignment in all_sum_assignments:
                    # Create full assignment dictionary
                    full_assignment = assignment_dict.copy()
                    full_assignment.update(dict(zip(sum_vars, sum_assignment)))
                    
                    # Compute potential and add to message value
                    message_value += from_clique['potential_func'](full_assignment)
            
            # Store message value for this sepset assignment
            message[tuple(assignment)] = message_value
        
        return message

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model using message passing.
        """
        # Make a copy of cliques to avoid modifying the original potentials
        working_cliques = [clique.copy() for clique in self.cliques]
        
        # Forward pass: from left to right
        for i in range(len(working_cliques) - 1):
            from_clique = working_cliques[i]
            to_clique = working_cliques[i + 1]
            
            # Compute message from from_clique to to_clique
            message = self.compute_message(from_clique, to_clique)
            self.messages[(from_clique['name'], to_clique['name'])] = message
            
            # Incorporate message into to_clique's potential
            sepset_vars = self.get_sepset_variables(from_clique, to_clique)
            
            # Create a new potential function for to_clique that includes the message
            old_potential_func = to_clique['potential_func']
            
            def new_potential_func(assignments, old_func=old_potential_func, msg=message, sep_vars=sepset_vars):
                # Get sepset assignment
                sep_assignment = tuple(assignments[var] for var in sep_vars)
                
                # Get message value for this sepset assignment
                message_value = msg.get(sep_assignment, 0.0)
                
                # Multiply with original potential
                return old_func(assignments) * message_value
            
            to_clique['potential_func'] = new_potential_func
        
        # Sum over all variables in the last clique to get Z-value
        last_clique = working_cliques[-1]
        last_vars = last_clique['variables']
        
        # Create all possible assignments for the last clique
        assignments_list = []
        for var in last_vars:
            if var.startswith('H_'):
                assignments_list.append(range(self.states_count))
            else:  # Observation variable
                t = int(var.split('_')[1])
                assignments_list.append([self.observation_sequence[t - 1]])  # observations are 0-indexed
        
        # Generate all combinations
        all_assignments = list(itertools.product(*assignments_list))
        
        z_value = 0.0
        for assignment in all_assignments:
            assignment_dict = dict(zip(last_vars, assignment))
            z_value += last_clique['potential_func'](assignment_dict)
        
        return z_value
    
    def triangulate_and_get_cliques(self):
        return self.cliques

    def get_junction_tree(self):
        return self.cliques, self.connections

    def compute_marginals(self):
        """
        Compute the marginal probabilities for all hidden variables using proper forward-backward message passing.
        Returns marginals in the order: H_1^1, H_1^2, ..., H_1^M, H_2^1, H_2^2, ..., H_T^M
        """
        M = self.factors_count
        T = self.num_observations
        K = self.states_count
        
        # Step 1: Forward pass (left to right) - incorporate messages as we go
        forward_messages = {}
        working_cliques_forward = [clique.copy() for clique in self.cliques]
        
        for i in range(len(working_cliques_forward) - 1):
            from_clique = working_cliques_forward[i]
            to_clique = working_cliques_forward[i + 1]
            
            # Compute message from from_clique to to_clique using current potentials
            message = self.compute_message(from_clique, to_clique)
            forward_messages[(from_clique['name'], to_clique['name'])] = message
            
            # INCORPORATE the message into the to_clique's potential
            sepset_vars = self.get_sepset_variables(from_clique, to_clique)
            old_potential_func = to_clique['potential_func']
            
            def new_potential_func(assignments, old_func=old_potential_func, msg=message, sep_vars=sepset_vars):
                sep_assignment = tuple(assignments[var] for var in sep_vars)
                message_value = msg.get(sep_assignment, 0.0)
                return old_func(assignments) * message_value
            
            to_clique['potential_func'] = new_potential_func
        
        # Step 2: Backward pass (right to left) - incorporate messages as we go
        backward_messages = {}
        working_cliques_backward = [clique.copy() for clique in self.cliques]
        
        for i in range(len(working_cliques_backward) - 1, 0, -1):
            from_clique = working_cliques_backward[i]
            to_clique = working_cliques_backward[i - 1]
            
            # Compute message from from_clique to to_clique using current potentials
            message = self.compute_message(from_clique, to_clique)
            backward_messages[(from_clique['name'], to_clique['name'])] = message
            
            # INCORPORATE the message into the to_clique's potential
            sepset_vars = self.get_sepset_variables(from_clique, to_clique)
            old_potential_func = to_clique['potential_func']
            
            def new_potential_func(assignments, old_func=old_potential_func, msg=message, sep_vars=sepset_vars):
                sep_assignment = tuple(assignments[var] for var in sep_vars)
                message_value = msg.get(sep_assignment, 0.0)
                return old_func(assignments) * message_value
            
            to_clique['potential_func'] = new_potential_func
        
        # Step 3: For each clique, compute the belief by combining:
        # - Original potential
        # - All forward messages (except those already incorporated)
        # - All backward messages (except those already incorporated)
        clique_beliefs = {}
        
        for clique in self.cliques:
            belief = {}
            clique_vars = clique['variables']
            
            # Create all possible assignments for this clique
            assignments_list = []
            for var in clique_vars:
                if var.startswith('H_'):
                    assignments_list.append(range(K))
                else:  # Observation variable
                    t = int(var.split('_')[1])
                    assignments_list.append([self.observation_sequence[t - 1]])
            
            all_assignments = list(itertools.product(*assignments_list))
            
            for assignment in all_assignments:
                assignment_dict = dict(zip(clique_vars, assignment))
                
                # Start with clique's original potential
                belief_value = clique['potential_func'](assignment_dict)
                
                # Multiply with forward messages from left that weren't incorporated
                clique_index = self.cliques.index(clique)
                for i in range(clique_index):
                    from_clique = self.cliques[i]
                    message_key = (from_clique['name'], clique['name'])
                    if message_key in forward_messages:
                        sepset_vars = self.get_sepset_variables(from_clique, clique)
                        sep_assignment = tuple(assignment_dict[var] for var in sepset_vars)
                        belief_value *= forward_messages[message_key].get(sep_assignment, 0.0)
                
                # Multiply with backward messages from right that weren't incorporated
                for i in range(clique_index + 1, len(self.cliques)):
                    from_clique = self.cliques[i]
                    message_key = (from_clique['name'], clique['name'])
                    if message_key in backward_messages:
                        sepset_vars = self.get_sepset_variables(from_clique, clique)
                        sep_assignment = tuple(assignment_dict[var] for var in sepset_vars)
                        belief_value *= backward_messages[message_key].get(sep_assignment, 0.0)
                
                belief[tuple(assignment)] = belief_value
            
            clique_beliefs[clique['name']] = belief
        
        # Step 4: Compute marginals for each hidden variable
        marginals = []
        
        for t in range(1, T + 1):
            for m in range(1, M + 1):
                var_name = f'H_{t}^{m}'
                marginal = [0.0] * K
                
                # Find cliques that contain this variable
                containing_cliques = [clique for clique in self.cliques if var_name in clique['variables']]
                
                if containing_cliques:
                    # Use the first clique that contains the variable
                    clique = containing_cliques[0]
                    belief = clique_beliefs[clique['name']]
                    
                    # Sum over all assignments where the variable has value k
                    for assignment, belief_value in belief.items():
                        assignment_dict = dict(zip(clique['variables'], assignment))
                        if var_name in assignment_dict:
                            k_value = assignment_dict[var_name]
                            marginal[k_value] += belief_value
                    
                    # Normalize
                    total = sum(marginal)
                    if total > 0:
                        marginal = [p / total for p in marginal]
                    else:
                        marginal = [1.0 / K] * K
                else:
                    marginal = [1.0 / K] * K
                
                marginals.append(marginal)
        
        return marginals

    def _safe_log(self, x):
        """Return log(x) with safe handling of 0 -> -inf."""
        if x is None:
            return -float('inf')
        try:
            if x <= 0:
                return -float('inf')
            return math.log(x)
        except:
            return -float('inf')

    def _safe_exp(self, lx):
        """Convert log back to probability (0 if -inf)."""
        if lx == -float('inf'):
            return 0.0
        return math.exp(lx)

    def _var_domain(self, var):
        """Return domain list for a variable name (H_* -> 0..K-1, O_* -> observed value only)."""
        if var.startswith('H_'):
            return list(range(self.states_count))
        # Observation var like 'O_3'
        t = int(var.split('_')[1])
        return [self.observation_sequence[t - 1]]

    def _all_assignments(self, var_list):
        """Yield tuples for all assignments to the given ordered var_list."""
        domains = [self._var_domain(v) for v in var_list]
        return itertools.product(*domains)

    def _project_tuple_from_assignment(self, full_tuple, full_vars, proj_vars):
        """
        Given an assignment tuple aligned with full_vars, return tuple aligned with proj_vars.
        (proj_vars order is preserved)
        """
        idx_map = {v: i for i, v in enumerate(full_vars)}
        return tuple(full_tuple[idx_map[v]] for v in proj_vars)

    def _topk_sum_of_lists(self, lists, k):
        """
        Given lists = [L1, L2, ...], each Li sorted in non-increasing order (largest first),
        return up to k tuples (sum, index_tuple) where index_tuple contains 0-based indices into each Li.
        This is the classic k-best sum-of-sorted-lists via max-heap.
        """
        if not lists:
            return [(0.0, tuple())]
        # If any list is empty -> impossible
        for L in lists:
            if not L:
                return []
        # value function
        def value(idx):
            s = 0.0
            for i, idv in enumerate(idx):
                s += lists[i][idv]
            return s

        start = tuple(0 for _ in lists)
        heap = [(-(value(start)), start)]
        seen = {start}
        out = []
        while heap and len(out) < k:
            negv, idx = heapq.heappop(heap)
            out.append((-negv, idx))
            # neighbors: increment each dimension
            for d in range(len(lists)):
                if idx[d] + 1 < len(lists[d]):
                    nb = list(idx); nb[d] += 1; nb = tuple(nb)
                    if nb not in seen:
                        seen.add(nb)
                        heapq.heappush(heap, (-(value(nb)), nb))
        return out

    # ----------------- k-best message computation -----------------
    def compute_kbest_messages(self, from_clique, to_clique, messages_in, k):
        """
        Compute k-best max-product message from `from_clique` -> `to_clique`.
        - messages_in: dict mapping child_clique_name -> that child's message dict
                       (child messages are dict: sepset_tuple -> list of entries {'score': logscore, 'c':..., 'child_ranks':...})
        Returns a dict: sepset_tuple -> list of entries.
        Each entry is a dict: {'score': logscore, 'c': c_assignment_tuple_aligned_with from_clique['variables'],
                               'child_ranks': {child_name: rank_index (1-based)} }
        """
        from_vars = from_clique['variables']
        sepset_vars = self.get_sepset_variables(from_clique, to_clique)
        rest_vars = [v for v in from_vars if v not in sepset_vars]

        # Precompute all possible sepset assignments
        sepset_domains = [self._var_domain(v) for v in sepset_vars]
        if sepset_domains:
            sepset_assignments = list(itertools.product(*sepset_domains))
        else:
            sepset_assignments = [tuple()]

        # Prepare output structure
        out_messages = {}

        # For mapping child names to their clique objects and message dicts
        child_names = list(messages_in.keys())

        # For each sepset assignment s, enumerate all possible assignments to the rest of the from_clique
        for s in sepset_assignments:
            candidates = []  # (logscore, c_assignment_tuple, child_ranks_map)
            # iterate over all assignments to rest_vars (if none, one empty tuple)
            for rest_assign in self._all_assignments(rest_vars):
                # Build full assignment aligned with from_vars order
                full_assign_values = []
                # create dict for from_clique potential evaluation
                assign_dict = {}
                # fill sepset vars
                for var, val in zip(sepset_vars, s):
                    assign_dict[var] = val
                # fill rest vars
                for var, val in zip(rest_vars, rest_assign):
                    assign_dict[var] = val
                # Build ordered tuple aligned with from_vars
                for v in from_vars:
                    full_assign_values.append(assign_dict[v])
                c_assignment_tuple = tuple(full_assign_values)

                # compute local potential log
                try:
                    pot_val = from_clique['potential_func'](assign_dict)
                except Exception:
                    # if potential function expects different types, fall back to 0
                    pot_val = 0.0
                log_base = self._safe_log(pot_val)

                # If there are child messages, for each child get the list of log-scores for the projected sepset
                child_lists = []
                child_proj_vars = []
                child_proj_keys = []
                skip = False
                for child_name in child_names:
                    child_msg = messages_in[child_name]
                    child_clique = self.get_clique_by_name(child_name)
                    proj_vars = self.get_sepset_variables(child_clique, from_clique)  # separator between child and this from_clique
                    child_proj_vars.append(proj_vars)
                    # compute projection s_h by looking up values from assign_dict
                    s_h = tuple(assign_dict[v] for v in proj_vars) if proj_vars else tuple()
                    child_proj_keys.append(s_h)
                    child_entries = child_msg.get(s_h, [])
                    if not child_entries:
                        skip = True
                        break
                    # extract list of log-scores (already stored as 'score')
                    child_lists.append([entry['score'] for entry in child_entries])
                if skip:
                    continue

                # Combine child-lists with topk sum (log-space addition)
                if child_lists:
                    combos = self._topk_sum_of_lists(child_lists, k)
                    # combos: list of (sum_child_logs, index_tuple) where index_tuple are 0-based indices into each child's list
                    for sum_child_log, idx_tuple in combos:
                        total_log = log_base + sum_child_log
                        # build child_ranks mapping (1-based indices)
                        child_ranks_map = {}
                        for ci, child_name in enumerate(child_names):
                            child_ranks_map[child_name] = idx_tuple[ci] + 1
                        candidates.append((total_log, c_assignment_tuple, child_ranks_map))
                else:
                    # no children: candidate is just log_base
                    candidates.append((log_base, c_assignment_tuple, {}))

            # keep top-k candidates for this sepset assignment
            if candidates:
                # get top-k by logscore
                topk = heapq.nlargest(k, candidates, key=lambda x: x[0])
                entries = []
                for logscore, c_tuple, child_ranks in topk:
                    entries.append({'score': logscore, 'c': c_tuple, 'child_ranks': child_ranks})
                out_messages[tuple(s)] = entries
            else:
                # No valid candidates -> empty list
                out_messages[tuple(s)] = []

        return out_messages

    # ----------------- Forward pass to compute messages for all edges -----------------
    def compute_top_k_assignments(self, k):
        """
        Run a forward pass computing k-best messages for every edge (left->right),
        then compute k-best assignments at the last clique (root).
        Returns:
           messages: dict mapping (from_name,to_name) -> message_dict (sepset_tuple -> list of entries)
           root_results: list of entries for the last clique (each entry is {'score', 'c', 'child_ranks'})
        """
        messages = {}
        # We'll process in left-to-right order as your junction tree list is already linear
        for i in range(len(self.cliques) - 1):
            from_clique = self.cliques[i]
            to_clique = self.cliques[i + 1]

            # children are neighbors of from_clique that are to the left (already processed)
            children = []
            if i - 1 >= 0:
                # the only possible left neighbor in this linear ordering is cliques[i-1]
                children.append(self.cliques[i - 1]['name'])

            # build messages_in mapping child_name -> messages[(child, from)]
            messages_in = {}
            for child_name in children:
                child_key = (child_name, from_clique['name'])
                if child_key not in messages:
                    # child message not computed (shouldn't happen in left-to-right order)
                    messages_in = {}
                    break
                messages_in[child_name] = messages[child_key]

            # compute k-best message for this edge
            m = self.compute_kbest_messages(from_clique, to_clique, messages_in, k)
            messages[(from_clique['name'], to_clique['name'])] = m

        # Now compute k-best assignments at the last clique (root)
        root = self.cliques[-1]
        # children of root are those to its left: index -1 neighbor
        children = []
        if len(self.cliques) >= 2:
            children.append(self.cliques[-2]['name'])

        # messages_in for root:
        messages_in_root = {}
        for child_name in children:
            messages_in_root[child_name] = messages[(child_name, root['name'])]

        # Root sepset is empty (we want assignments over entire root clique)
        # We'll reuse compute_kbest_messages with a dummy 'to_clique' with no intersection to root.
        # Create a fake to_clique with empty variables (so sepset_vars = [])
        fake_to = {'name': '__ROOT_TARGET__', 'variables': [], 'type': 'fake'}
        root_results = self.compute_kbest_messages(root, fake_to, messages_in_root, k).get(tuple(), [])

        return messages, root_results

    # ----------------- Backtracking to full assignments -----------------
    def backtrack_kbest_assignments(self, messages, root_results, k):
        """
        Given the forward messages and the list of root k-best entries, reconstruct full assignments
        for each root entry (recursively following child backpointers).
        Returns list of tuples (assignment_dict, probability) sorted by probability descending.
        """
        def reconstruct_from_entry(clique_name, entry):
            """
            Recursively reconstruct all variable assignments under this clique entry.
            Returns a dict mapping var_name -> value (for the subtree rooted at this clique).
            """
            assign = {}
            clique = self.get_clique_by_name(clique_name)
            c_vars = clique['variables']
            c_tuple = entry['c']
            # assign clique variables
            for idx, var in enumerate(c_vars):
                assign[var] = c_tuple[idx]

            # for each child indicated in child_ranks, follow that child's chosen entry
            for child_name, rank in entry['child_ranks'].items():
                # locate child's message dict
                key = (child_name, clique_name)
                child_msg_dict = messages.get(key, {})
                # compute separator projection between child and this clique
                child_clique = self.get_clique_by_name(child_name)
                sep_vars = self.get_sepset_variables(child_clique, clique)
                # build sepset tuple from current assign
                sep_tuple = tuple(assign[v] for v in sep_vars) if sep_vars else tuple()
                # get the child entry (rank is 1-based)
                child_entries = child_msg_dict.get(sep_tuple, [])
                if not child_entries:
                    # inconsistent / missing - skip
                    continue
                child_entry = child_entries[rank - 1]
                # recursively reconstruct
                child_assign = reconstruct_from_entry(child_name, child_entry)
                assign.update(child_assign)
            return assign

        results = []
        for entry in root_results[:k]:
            # entry is {'score': logscore, 'c': tuple, 'child_ranks': {...}}
            full_assign = reconstruct_from_entry(self.cliques[-1]['name'], entry)
            # compute joint probability (from factor arrays, not mutated clique potentials)
            prob = self.compute_assignment_probability(full_assign)
            results.append((full_assign, prob, entry['score']))

        # sort by prob descending
        results.sort(key=lambda x: x[1], reverse=True)
        # return (assignment, prob) pairs
        return [(r[0], r[1]) for r in results[:k]]

    # ----------------- High-level compute_top_k -----------------
    def compute_top_k(self, k=2):
        """
        Top-k assignments (full hidden-state assignments).
        Returns list of dicts {"assignment": {...}, "probability": prob}
        """
        messages, root_results = self.compute_top_k_assignments(k)
        topk = self.backtrack_kbest_assignments(messages, root_results, k)
        output = []
        z_value = self.get_z_value()
        for assignment, prob in topk:
            # ensure observation variables are present
            current_assignment = []
            for t in range(1, self.num_observations + 1):
                for m in range(1, self.factors_count + 1):
                    current_assignment.append(assignment.get(f'H_{t}^{m}', 0))
            output.append({"assignment": current_assignment, "probability": prob / z_value if z_value > 0 else 0.0})
        return output

    # ----------------- Compute joint probability from base factors -----------------
    def compute_assignment_probability(self, assignment):
        """
        Compute the joint probability of a full assignment over all variables
        using the original factor tables (self.normalized_state_potentials and self.normalized_transition_potentials).
        This avoids using any modified clique potential functions.
        """
        prob = 1.0
        M = self.factors_count
        T = self.num_observations
        K = self.states_count

        # state factor potentials (per time t)
        for t in range(1, T + 1):
            # collect hidden values H_t^1 ... H_t^M
            h_vals = []
            for m in range(1, M + 1):
                var = f'H_{t}^{m}'
                if var in assignment:
                    h_vals.append(int(assignment[var]))
                else:
                    # if missing, treat as 0 (shouldn't happen if reconstruction is correct)
                    h_vals.append(0)
            o_val = int(self.observation_sequence[t - 1])
            # compute index consistent with create_potential_function
            # NOTE: we keep the same indexing formula used earlier
            index = o_val
            for m, h_val in enumerate(h_vals):
                index += h_val * (K ** (M - m))
            # guard index bounds
            if 0 <= index < len(self.normalized_state_potentials):
                prob *= self.normalized_state_potentials[index]
            else:
                prob *= 0.0

        # transition potentials (for t=1..T-1 and for each chain n)
        for t in range(1, T):
            for n in range(self.factors_count):
                var1 = f'H_{t}^{n+1}'
                var2 = f'H_{t+1}^{n+1}'
                v1 = int(assignment.get(var1, 0))
                v2 = int(assignment.get(var2, 0))
                idx = v1 * K + v2
                prob *= self.normalized_transition_potentials[n][idx]

        return prob


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
    evaluator.write_output('Output.json')