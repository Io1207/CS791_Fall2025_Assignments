import math
import json
import itertools
from collections import defaultdict

########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################

def logsumexp(vals):
    m=max(vals)
    return m+math.log(sum(math.exp(v - m) for v in vals))

def log_normalize(log_vec):
    lse=logsumexp(log_vec)
    return [v - lse for v in log_vec]

def normalize(vec):
    s=sum(vec)
    if s==0:
        K=len(vec)
        return [1.0/K]*K
    return [v/s for v in vec]

class Inference:
    def __init__(self, data, max_iters=2000, tol=1e-12, damping=1.0):
        self.F=data['Factors_Count']
        self.K=data['State_Count']
        self.T=data['Number of Observations']
        self.obs=data['Observation Sequence']
        self.state_factor_potential=data['State_Factor_Potentials']
        self.transition_potentials=list(data['Transition Potentials'].values())
        self.observation_sequence=list(data['Observation Sequence'])

        #normalising transition potentials 
        raw_trans=list(data['Transition Potentials'].values())
        self.trans=[]
        for flat in raw_trans:
            newflat=[]
            for row in range(self.K):
                row_start=row*self.K
                row_block=flat[row_start: row_start+self.K]
                row_sum=sum(row_block)
                if row_sum==0:
                    newflat.extend([1.0/self.K]*self.K)
                else:
                    newflat.extend([v/row_sum for v in row_block])
            self.trans.append(newflat)

        #normalizing emission table across all transitions
        raw_emis=data['State_Factor_Potentials']
        self.emis_map={}
        self.assigns=list(itertools.product(range(self.K), repeat=self.F))
        for a_idx, assign in enumerate(self.assigns):
            start=a_idx*self.K
            block=raw_emis[start:start+self.K]
            s=sum(block)
            if s==0:
                norm_block=[1.0/self.K]*self.K
            else:
                norm_block=[v/s for v in block]
            for o in range(self.K):
                self.emis_map[(assign, o)]=norm_block[o]

        #building graph
        self.build_graph()

        self.var_to_factors=defaultdict(list)
        for fac in self.factor_nodes:
            for v in fac['vars']:
                self.var_to_factors[v].append(fac['id'])

        #initializing messages
        #changes from uniform to logspace to minimise errors
        log_uniform=[-math.log(self.K)]*self.K
        self.factor_to_var={}
        self.var_to_factor={}
        for fac in self.factor_nodes:
            for v in fac['vars']:
                #non logspace messages
                #self.factor_to_var[(fac['id'], v)]=[1.0/self.K]*self.K
                #self.var_to_factor[(v, fac['id'])]=[1.0/self.K]*self.K
                self.factor_to_var[(fac['id'], v)]=log_uniform[:]
                self.var_to_factor[(v, fac['id'])]=log_uniform[:]

        self.max_iters=max_iters
        self.tol=tol
        self.damping=damping

    def build_graph(self):
        self.factor_nodes=[]
        for t in range(self.T):
            fid=len(self.factor_nodes)
            self.factor_nodes.append({
                'type': 'emis',
                'vars': [(t, f) for f in range(self.F)],
                't': t,
                'id': fid
            })

        for t in range(1, self.T):
            for f in range(self.F):
                fid=len(self.factor_nodes)
                self.factor_nodes.append({
                    'type': 'trans',
                    'vars': [(t-1, f), (t, f)],
                    't': t,
                    'f': f,
                    'id': fid
                })

    # subroutine 1 in run
    def loopy_belief_propagate_to_variable(self, fac, fid):
        """
            compute factor -> variable messages for 1
            factor (returns dict of {(fid,var): message})
        """
        new_msgs={}
        if fac['type']=='trans':
            (v_prev, v_curr)=fac['vars']
            f_idx=fac['f']
            trans_flat=self.trans[f_idx]

            """
                Update Rule
                m_{f->v_prev}(xp)=sum_{xc} phi(xp,xc)*m_{v_curr->f}(xc)
            """
            msg_prev=[]
            inc_curr=self.var_to_factor[(v_curr, fid)]
            # msg_prev=[0.0]*self.K
            for xp in range(self.K):
                # s=0.0
                terms=[]
                for xc in range(self.K):
                    # s +=trans_flat[xp*self.K+xc]*inc_curr[xc]
                    terms.append(math.log(trans_flat[xp*self.K+xc])+inc_curr[xc])
                # msg_prev[xp]=s
                msg_prev.append(logsumexp(terms))
            new_msgs[(fid, v_prev)]=log_normalize(msg_prev)

            """ 
                Update Rule:
                m_{f->v_curr}(xc)=sum_{xp} phi(xp,xc)*m_{v_prev->f}(x) 
            """
            msg_curr=[]
            inc_prev=self.var_to_factor[(v_prev, fid)]
            # msg_curr=[0.0]*self.K
            for xc in range(self.K):
                # s=0.0
                terms=[]
                for xp in range(self.K):
                    # s+=trans_flat[xp*self.K+xc]*inc_prev[xp]
                    terms.append(math.log(trans_flat[xp*self.K+xc])+inc_prev[xp])
                #msg_curr[xc]=s
                msg_curr.append(logsumexp(terms))
            new_msgs[(fid, v_curr)]=log_normalize(msg_curr)

        elif fac['type']=='emis':
            t=fac['t']
            obs_t=self.obs[t]
            vars_list=fac['vars']
            incoming={v: self.var_to_factor[(v, fac['id'])] for v in vars_list}
            assigns=list(itertools.product(range(self.K), repeat=len(vars_list)))

            # for each target variable, sum over other variables (excluding target's incoming)
            for i, v in enumerate(vars_list):
                msg=[-math.inf]*self.K
                for assign in assigns:
                    phi=math.log(self.emis_map[(assign, obs_t)])
                    prod_except_i=0.0
                    for j, vj in enumerate(vars_list):
                        if j==i:
                            continue
                        prod_except_i+=incoming[vj][assign[j]]
                    val=phi+prod_except_i
                    ai=assign[i]
                    msg[ai]=logsumexp([msg[ai], val])
                new_msgs[(fac['id'], v)]=log_normalize(msg)
        return new_msgs
        raise NotImplementedError()

    #subroutine 2 in the main loop
    def loopy_belief_propagate_to_factor(self):
        """
            compute variable -> factor messages for all 
            variables (returns dict of {(var,fid): message})
        """
        new_msgs={}
        for v in self.var_to_factors:
            facs=self.var_to_factors[v]
            for fid in facs:
                prod_msg=[0.0]*self.K
                for other_f in facs:
                    if other_f==fid:
                        continue
                    inc=self.factor_to_var[(other_f, v)]
                    for k in range(self.K):
                        prod_msg[k]+=inc[k]
                new_msgs[(v, fid)]=log_normalize(prod_msg)
        return new_msgs
        raise NotImplementedError()

    def factor_in_fhmm(self):
        """
            runs the iteration in lbp
            subroutine 3 in run, technically the main loop itself
        """        
        for _ in range(self.max_iters):
            max_delta=0.0

            # factors to variables
            new_factor_to_var={}
            for fac in self.factor_nodes:
                fid=fac['id']
                msgs=self.loopy_belief_propagate_to_variable(fac, fid)
                new_factor_to_var.update(msgs)

            # damping, assign
            for key, new_msg in new_factor_to_var.items():
                old=self.factor_to_var[key]

                old_prob=[math.exp(v) for v in old]
                new_prob=[math.exp(v) for v in new_msg]

                damped_prob=[
                    (1-self.damping)*old_prob[i]+self.damping*new_prob[i]
                    for i in range(self.K)
                ]
                damped_log=[math.log(p) for p in damped_prob]
                damped=log_normalize(damped_log)

                self.factor_to_var[key]=damped
                max_delta=max(max_delta,max(abs(damped[i]-old[i]) for i in range(self.K)))

            # variables to factor
            new_var_to_factor=self.loopy_belief_propagate_to_factor()
            for key, new_msg in new_var_to_factor.items():
                old=self.var_to_factor[key]

                old_prob=[math.exp(v) for v in old]
                new_prob=[math.exp(v) for v in new_msg]

                damped_prob=[
                    (1-self.damping)*old_prob[i]+self.damping*new_prob[i]
                    for i in range(self.K)
                ]
                damped_log=[math.log(p) for p in damped_prob]
                damped=log_normalize(damped_log)

                self.var_to_factor[key]=damped
                max_delta=max(max_delta,max(abs(damped[i]-old[i]) for i in range(self.K)))

            if max_delta<self.tol:
                break
        return True
        raise NotImplementedError()

    # loop controller
    def compute_marginals(self):
        """
            call facor_in_hmm() which in turn calls the propagation functions
            returns marginals
        """
        self.factor_in_fhmm()
        beliefs={}
        for v in self.var_to_factors:
            prod=[0.0]*self.K
            for fid in self.var_to_factors[v]:
                inc=self.factor_to_var[(fid, v)]
                for k in range(self.K):
                    prod[k]+=inc[k]
            lse=logsumexp(prod)
            beliefs[v]=[math.exp(val - lse) for val in prod]

        marginals=[]
        for t in range(self.T):
            for f in range(self.F):
                marginals.append(beliefs[(t, f)])
        return marginals


########################################################################

# Do not change anything below this line

########################################################################



class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data=json.load(file)

    def get_output(self):
        output=[]
        i=1
        for case in self.data:
            LBPinf=Inference(case['Input'], damping=0.9, max_iters=2000, tol=1e-12)
            marginals=LBPinf.compute_marginals()
            output.append({
                "Testcase ":str(i),
                'Marginals': marginals
                })
            i=i+1
        self.output=output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__=='__main__':
    # evaluator=Get_Input_and_Check_Output('./Q2/TestCases.json')
    evaluator=Get_Input_and_Check_Output('TestCases.json')
    evaluator.get_output()
    # evaluator.write_output('./Q2/OutputQ2.json')
    evaluator.write_output('OutputQ2.json')