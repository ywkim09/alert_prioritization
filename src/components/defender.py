import json
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import sys
import torch
EPS = np.finfo(float).eps
from scipy.special import entr
import math
import time
try:
    from src.util.auxiliaries import checkAttackerType, convKeyInt, convertDec2Bin, powerSet
    from src.components.ids import IDS
    from src.components.graph import Graph
except:
    import os
    p = os.path.abspath('.')
    sys.path.insert(1, p)
    from src.components.ids import IDS
    from src.components.graph import Graph
    from src.util.auxiliaries import checkAttackerType, convKeyInt, convertDec2Bin, powerSet



class Defender:
    def __init__(self, ids: IDS, graph : Graph, choicePath, successPath, discount = 0.95, weight = 0.5, testDefenses = 0, numSamples= 1200):
        self.defenses = testDefenses
        
        self.graph = graph
        self.ids = ids
        self.effect = self.graph.effect
        self.type2int = self.ids.type2int
        self.feasibleStates = self.graph.feasibleStates
        self.feasibleExploits = self.graph.feasibleExploits
        self.idxFeasibleStates = self.graph.idxFeasibleStates
        self.numCondition = self.graph.numCondition
        self.idsTPMatrix = self.ids.idsTPMatrix
        self.idsFPMatrix = self.ids.idsFPMatrix.T
        
        self.numAttTypes =  self.idsFPMatrix.shape[-1]
        self.numExploit = self.graph.numExploit
        self.preCondition = self.graph.preCondition
        self.postCondition = self.graph.postCondition
        self.goalStates = 2**self.graph.goalState - 1
        self.numAlerts = self.ids.idsTPRate.shape[0]

        self.choiceRate = np.zeros((self.numAttTypes, self.graph.numExploit, 2)) 
        with open(choicePath, 'r') as f:
            choiceRate = json.load(f)
        choiceRate = convKeyInt(choiceRate)
        for k,v in choiceRate.items():
            self.choiceRate[self.type2int[k],:,:] = np.array([v[k2] for k2 in sorted(v.keys())])

        self.successRate = np.zeros((self.numAttTypes, self.graph.numExploit))
        with open(successPath, 'r') as f:
            successRate = json.load(f)
        successRate = convKeyInt(successRate)
        for k,v in successRate.items():
            self.successRate[self.type2int[k],:] = [v[k2] for k2 in sorted(v.keys())]

        self.belief = np.zeros([len(self.graph.feasibleStates), self.numAttTypes])
        self.belief[0,:] = 1/self.numAttTypes
        self.q = np.eye(self.numAttTypes)
        self.weight = weight
        self.discount = discount
        self.defenseActions = list(range(len(self.effect.keys()) + 1))
        self.c_u = np.array([0.0] +[0.25]*len(self.effect.keys()))
        self.c_s = np.zeros([self.belief.shape[0], self.numAttTypes])
        self.c_s[np.where(self.feasibleStates>=self.goalStates[0]),:]  = 1
        self.c_il = np.zeros(self.belief.shape)
        self.numSamples = numSamples
        self.beliefRaw = np.zeros([len(self.graph.feasibleStates), self.numAttTypes])
        self.beliefRaw[0,:] = 1/self.numAttTypes
        self.availExploitLst = []
        for s in self.feasibleStates:
            availExploit = self.graph.checkAvailExploit(s)
            self.availExploitLst.append(availExploit)
        self.iBins = convertDec2Bin(self.feasibleStates, self.numCondition)

    def cost(self, belief):
        cost = (np.tile(np.expand_dims(self.weight * self.c_s, axis= -1), len(self.c_u))
                + (1-self.weight) * self.c_u) * np.expand_dims(belief, axis=-1) 
        return cost

    def build_p_ijl(self):
        p_ijl = np.zeros([self.feasibleStates.shape[0], self.feasibleStates.shape[0], self.numAttTypes])
        for iIdx, i in enumerate(self.feasibleStates):
            possibleExploit, blockedExploit = self.graph.checkPossibleTransition(i)
            iBin = convertDec2Bin(i, self.numCondition)
            for exploits in powerSet(possibleExploit):
                v1= np.array(exploits, dtype=int)
                v0= np.array(np.setdiff1d(possibleExploit, v1), dtype=int)
                
                binNextPossStates = (iBin + np.sum(self.postCondition[v1], axis=0)) > 0
                j = binNextPossStates.dot(2**np.arange(binNextPossStates.size))
                
                probChoice = np.ones([self.numAttTypes, self.numExploit])
                probSuccess = np.ones([self.numAttTypes, self.numExploit])
                
                probChoice[:, possibleExploit] = self.choiceRate[:, possibleExploit, 0]
                probSuccess[:, possibleExploit] = self.successRate[:, possibleExploit]
                
                jProbCompromise = probChoice * probSuccess
                jProbCompromise[:, v0] = 1 - jProbCompromise[:, v0] 

                p_ij = np.prod(jProbCompromise, axis = 1)
                
                jIdx = self.idxFeasibleStates[j] # np.where(self.feasibleStates == j )
                p_ijl[jIdx,iIdx,:] += p_ij
        return p_ijl

    def r_ijkl(self, alerts_n, p_ijl):
        r_ijkl = np.zeros([p_ijl.shape[0], p_ijl.shape[0], self.numAttTypes])
        for iIdx, i in enumerate(self.feasibleStates):
            iBin = convertDec2Bin(i, self.numCondition)         
            availExploits = self.graph.checkAvailExploit(i)
            availExploits = availExploits[availExploits>=0]
            blockedExploits = self.graph.checkBlockedExploit(availExploits)
            possExploits = np.setdiff1d(availExploits, blockedExploits)
            choiceRate = np.ones([self.numExploit, self.numAttTypes])
            choiceRate[possExploits, :] = self.choiceRate[:, possExploits, 0].T
            choiceRate[blockedExploits, :] = self.choiceRate[:, blockedExploits, 1].T
            for chosenExploits in powerSet(availExploits):
                sum_beta = np.zeros([p_ijl.shape[0], self.numAttTypes])
                chosenExploits = np.array(chosenExploits, dtype=int)
                notChosenExploits = np.setdiff1d(availExploits, chosenExploits) 
                negativeAlerts = np.setdiff1d(np.arange(self.numAlerts), alerts_n)
                
                probAlert = 1 - (np.prod(1 - self.idsTPMatrix[alerts_n, :, :][:, chosenExploits, :], axis=1) * (1 - self.idsFPMatrix[alerts_n]))
                probNoAlert = np.prod(1 - self.idsTPMatrix[negativeAlerts, :, :][:, chosenExploits, :], axis=1) * (1 - self.idsFPMatrix[negativeAlerts])
                probCurrAlert = np.prod(np.vstack([probAlert, probNoAlert]), axis = 0) #Probability of the current alert by chosen exploits
                alpha = np.prod(choiceRate[chosenExploits, :], axis=0) * np.prod(1 - choiceRate[notChosenExploits, :], axis=0)
                for successExploits in powerSet(chosenExploits):
                    v1= np.array(successExploits, dtype=int)
                    # v0= np.array(np.setdiff1d(possExploits, v1), dtype=int)
                    v0= np.array(np.setdiff1d(chosenExploits, v1), dtype=int)

                    beta = np.ones([self.numAttTypes, self.numExploit])
                    beta[:, v1] = self.successRate[:, v1]
                    beta[:, v0] = 1 - self.successRate[:, v0] 

                    binNextPossStates = (iBin + np.sum(self.postCondition[v1], axis=0)) > 0
                    j = binNextPossStates.dot(2**np.arange(binNextPossStates.size))
                    jIdx = self.idxFeasibleStates[j] #np.where(self.feasibleStates == j )
                    sum_beta[jIdx, :] += np.prod(beta, axis=1)
                r_ijkl[:,iIdx,:] += probCurrAlert * sum_beta * alpha
        r_ijkl[p_ijl>0] /= p_ijl[p_ijl>0]
        return r_ijkl

    def updateBelief(self, alerts_n, chosenExploit, trueAlerts):
        p_ijl= self.build_p_ijl()
        r_ijkl = self.r_ijkl(alerts_n, p_ijl)

        p_jm = np.sum(np.dot(self.beliefRaw * p_ijl, self.q), axis = 1)
        r_jk = np.sum(np.sum(self.beliefRaw * r_ijkl, axis = -1), axis=-1)

        numeratorRaw = p_jm.T * r_jk
        self.beliefRaw = numeratorRaw.T / np.sum(numeratorRaw)
        return {"Raw": self.beliefRaw}
        
class MyDefender(Defender):
    def __init__(self, ids: IDS, graph : Graph, choicePath, successPath, policies, epsilon, seed, state_threshold = 0.0, exploit_threshold = 0.0, ib = 1, testDefense = 0, 
                discount = 0.95, weight = 0.5, testDefenses = 0, numSamples= 1200, omega = [1], RL_policy = None):
        super().__init__(ids, graph, choicePath, successPath, discount, weight, testDefenses, numSamples)
        self.w_list = omega
        self.RL_policy = RL_policy
        self.policies = policies
        self.epsilon = epsilon
        self.beliefDict ={}
        self.step = 0
        for policy in policies:
            self.beliefDict[policy] = {}
            for omega in self.w_list:
                if policy=="Raw":
                    self.beliefDict[policy][omega[0]] = np.ones([len(self.graph.feasibleStates), self.numAttTypes])
                    self.beliefDict[policy][omega[0]][:] = self.epsilon
                    self.beliefDict[policy][omega[0]][0,:] = 1 / self.numAttTypes
                elif  policy == "RL":
                    self.beliefDict[policy][omega[0]] = np.ones([len(self.graph.feasibleStates), self.numAttTypes])
                    self.beliefDict[policy][omega[0]][:] = self.epsilon
                    self.beliefDict[policy][omega[0]][0,:] = 1 / self.numAttTypes
                    
                    # self.beliefDict["RL"]["untrained"] = np.ones([len(self.graph.feasibleStates), self.numAttTypes])
                    # self.beliefDict["RL"]["untrained"][:] = self.epsilon
                    # self.beliefDict["RL"]["untrained"][0,:]  = 1 / self.numAttTypes
                else:
                    self.beliefDict[policy][omega[0]] = np.ones([len(self.graph.feasibleStates), self.numAttTypes])
                    self.beliefDict[policy][omega[0]][:] = self.epsilon
                    self.beliefDict[policy][omega[0]][0,:] = 1 / self.numAttTypes
        self.beliefRL = np.ones([len(self.graph.feasibleStates), self.numAttTypes]) * self.epsilon
        self.beliefRL[:] = self.epsilon
        self.beliefRL[0,:] = 1/self.numAttTypes
        self.ib_each_human = {}
        self.ib = {}
        for p in policies: 
            self.ib_each_human[p] = np.array([ib])
            self.ib[p] = self.ib_each_human[p].cumsum()
        self.ib_each_human['All'][:] = 999999
        self.ib['All'][:] = 999999
        self.policy = policy
        self.investigationList = []
        self.investigationList_Y = []
        self.exploits_for_i = {}
        self.state_threshold = state_threshold
        self.dynamics = self.build_p_ijl()
        self.cnt_non_zero = 0
        self.state_e_powerset = {}
        self.prob_lst = []
        for iIdx, iBin in enumerate(self.iBins):
            choiceRate = np.ones([self.numExploit, self.numAttTypes])
            availExploits = self.graph.checkAvailExploit_bin(iBin)
            availExploits = availExploits[availExploits>=0]
            blockedExploits = self.graph.checkBlockedExploit(availExploits)
            possExploits = np.setdiff1d(availExploits, blockedExploits)
            choiceRate[possExploits, :] = self.choiceRate[:, possExploits, 0].T
            choiceRate[blockedExploits, :] = self.choiceRate[:, blockedExploits, 1].T
            for E_t in powerSet(availExploits):
                not_E_t = np.setdiff1d(availExploits, E_t)
                alpha = np.prod(choiceRate[E_t, :], axis=0) * np.prod(1 - choiceRate[not_E_t, :], axis=0)
                self.prob_lst.append(np.max(alpha))
        # self.exploit_threshold = sorted(self.prob_lst)[round(len(self.prob_lst)*exploit_threshold)]
        self.exploit_threshold = exploit_threshold
        for iIdx, iBin in enumerate(self.iBins):
            self.state_e_powerset[iIdx] = []
            choiceRate = np.ones([self.numExploit, self.numAttTypes])
            availExploits = self.graph.checkAvailExploit_bin(iBin)
            availExploits = availExploits[availExploits>=0]
            blockedExploits = self.graph.checkBlockedExploit(availExploits)
            possExploits = np.setdiff1d(availExploits, blockedExploits)
            choiceRate[possExploits, :] = self.choiceRate[:, possExploits, 0].T
            choiceRate[blockedExploits, :] = self.choiceRate[:, blockedExploits, 1].T
            for E_t in powerSet(availExploits):
                not_E_t = np.setdiff1d(availExploits, E_t)
                alpha = np.prod(choiceRate[E_t, :], axis=0) * np.prod(1 - choiceRate[not_E_t, :], axis=0)
                if np.max(alpha) >= self.exploit_threshold:
                    self.state_e_powerset[iIdx].append(E_t)
        self.cnt_zero = 0
        self.numState = len(self.state_e_powerset.keys())
        self.time_dict = {}
        for p in policies:
            self.time_dict[p] = {}
            for omega in self.w_list:
                # self.time_dict[p][omega] = 0
                self.time_dict[p][omega[0]] = 0
        self.omega_random_state = {}
        self.mistake_omega_random_state = {}
        self.policy_omega_random_state = {}
        for omega in self.w_list:
            # self.omega_random_state[omega] = np.random.RandomState(seed + int(omega*10**7))
            self.omega_random_state[omega[0]] = np.random.RandomState(seed + int(omega[0]*10**7))
        for omega in self.w_list:
            # self.mistake_omega_random_state[omega] = np.random.RandomState(seed + int(omega*10**7))
            self.mistake_omega_random_state[omega[0]] = np.random.RandomState(seed + int(omega[0]*10**7))
        for p in policies:
            self.policy_omega_random_state[p] = {}
            for omega in self.w_list:
                for o in omega:
                    self.policy_omega_random_state[p][o] = np.random.RandomState(seed + 1 + int(o*10**7))
    def randomPolicy(self, y_t, trueAlerts, omega, security_anlayst):
        p = "Random"
        if len(y_t) != 0:
            num_pos_alert = len(y_t)
            shuffle_idx = self.policy_omega_random_state[p][omega].choice(num_pos_alert, size=num_pos_alert, replace=False)
            shuffle_y_t = y_t[shuffle_idx]
            investIdx = shuffle_y_t[:self.ib[p][security_anlayst]]
            investResult = trueAlerts[investIdx]        
        else:
            investIdx = []
            investResult = []
        return (np.array(investIdx, dtype=np.int8).reshape(-1), np.array(investResult, dtype=np.bool).reshape(-1))

    def lowFPPolicy(self, y_t, trueAlerts, belief, omega, security_anlayst):
        p = "MinFP"
        if len(y_t) != 0:
            num_pos_alert = len(y_t)
            belief_type = belief.sum(axis=0)
            rate = (self.idsFPMatrix * belief_type).sum(axis=1)[y_t]
            fp_rates_proportion = np.divide(rate, rate.sum(), out=np.zeros_like(rate), where=rate.sum()!=0)
            all_equal = np.all(fp_rates_proportion == fp_rates_proportion[0])
            if all_equal is True:
                shuffle_idx = self.policy_omega_random_state[p][omega].choice(num_pos_alert, size=num_pos_alert, replace=False)
                shuffle_y_t = y_t[shuffle_idx]
                # self.policy_omega_random_state[p][omega].shuffle(y_t)
                investIdx = shuffle_y_t[:self.ib[p][security_anlayst]]
                investResult = trueAlerts[investIdx]        
            else:
                investIdx = y_t[fp_rates_proportion.argsort()[:self.ib[p][security_anlayst]]]
                investResult = trueAlerts[investIdx]        
        else:
            investIdx = []
            investResult = []
        return (np.array(investIdx, dtype=np.int8).reshape(-1), np.array(investResult, dtype=np.bool).reshape(-1))

    def maxEntropyPolicy(self, alerts, trueAlerts, belief, omega, security_anlayst):
        p = "MaxEntropy"
        if len(alerts) != 0:
            entropies_wo_min = np.zeros([len(alerts)])
            entropies_gyuri = np.zeros([len(alerts)])
            investIdx = []
            investResult = []
            policy = np.zeros([2, len(alerts), self.feasibleStates.shape[0], self.numAttTypes])
            for aIdx, a in enumerate(alerts):   
                states1 = []
                states2 = []
                raw_idx = belief.sum(axis=1).argsort()[::-1]
                belief_cumsum = belief.sum(axis=1)[raw_idx].cumsum()
                flt_state_idx = raw_idx[belief_cumsum <= (1-self.state_threshold)]
                if len(flt_state_idx)==0:
                    flt_state_idx = raw_idx[:1]
                for iIdx, iBin in zip(flt_state_idx, self.iBins[flt_state_idx]):
                    e_s_i = self.graph.checkAvailExploit_bin(iBin)
                    e_s_i = e_s_i[e_s_i>=0]
                    tpProbSum, fpProbSum, tp_ProbSum, fp_ProbSum = self.tpAndfpProb(iBin, iIdx, e_s_i, a, omega)
                    states1.append(tp_ProbSum)
                    states2.append(fp_ProbSum)
                    policy[0, aIdx, iIdx] = tpProbSum
                    policy[1, aIdx, iIdx] = fpProbSum
                states1 =np.array(states1).transpose(1,0,2)* belief[flt_state_idx]
                states2 =np.array(states2).transpose(1,0,2)* belief[flt_state_idx]
                states = np.array([states1, states2])
                entropies_gyuri[aIdx] = entr(states).sum()/np.log(2)
            policy = policy * belief
            trans_policy = policy.transpose(1,0,2,3)
            entropies_w_min = np.zeros([len(alerts), 2])
            for i, each_alert in enumerate(trans_policy):
                entropies_wo_min[i] = entr(each_alert).sum()/np.log(2)
                for j, true_false in enumerate(each_alert):
                    entropies_w_min[i,j] = entr(true_false).sum()/np.log(2)
            investIdx = alerts[entropies_gyuri.argsort()[-self.ib[p][security_anlayst]:]]
            investResult = trueAlerts[investIdx]
                # investIdx = alerts[entropies_gyuri.argsort()[-self.ib:]]
                # investResult = trueAlerts[investIdx]                          
        else:
            investIdx = []
            investResult = []
        return (np.array(investIdx, dtype=np.int8).reshape(-1), np.array(investResult, dtype=np.bool).reshape(-1))
     
    def tpAndfpProb(self, iBin, iIdx, e_s_i, a, omega):
        tpProbSum, fpProbSum = 0, 0
        tp_or_tp_ProbSum = np.zeros([len(self.feasibleStates), self.numAttTypes])
        denominator = 0
        for chosenExploits in self.state_e_powerset[iIdx]:
            chosenExploits                 = np.array(chosenExploits, dtype=np.int8)
            notChosenExploits              = np.setdiff1d(e_s_i, chosenExploits)

            probChoice                     = np.ones([self.numExploit, self.numAttTypes])
            blockedExploits                = self.graph.checkBlockedExploit(e_s_i)
            possExploits                   = np.setdiff1d(e_s_i, blockedExploits)
            
            probChoice[possExploits, :]    = self.choiceRate[:, possExploits, 0].T
            probChoice[blockedExploits, :] = self.choiceRate[:, blockedExploits, 1].T
            probChoiceRate                 = np.prod(probChoice[chosenExploits, :], axis=0) * np.prod(1 - probChoice[notChosenExploits, :], axis=0)

            probAlert                      = 1 - (np.prod(1 - self.idsTPMatrix[a, chosenExploits, :], axis=0) * (1 - self.idsFPMatrix[a]))
            denominator += probChoiceRate * probAlert
        
        for chosenExploits in powerSet(e_s_i):
            sum_beta = np.zeros([len(self.feasibleStates), self.numAttTypes])
            for v1 in powerSet(chosenExploits):
                v1= np.array(v1, dtype=int)
                v0= np.array(np.setdiff1d(e_s_i, v1), dtype=int)

                beta = np.ones([self.numAttTypes, self.numExploit])
                beta[:, v1] = self.successRate[:, v1]
                beta[:, v0] = 1 - self.successRate[:, v0] 

                binNextPossStates = (iBin + np.sum(self.postCondition[v1], axis=0)) > 0
                j = binNextPossStates.dot(2**np.arange(binNextPossStates.size))
                jIdx = self.idxFeasibleStates[j] 
                sum_beta[jIdx, :] += np.prod(beta, axis=1)
            
            chosenExploits                 = np.array(chosenExploits, dtype=np.int8)
            notChosenExploits              = np.setdiff1d(e_s_i, chosenExploits)

            probChoice                     = np.ones([self.numExploit, self.numAttTypes])
            blockedExploits                = self.graph.checkBlockedExploit(e_s_i)
            possExploits                   = np.setdiff1d(e_s_i, blockedExploits)
            
            probChoice[possExploits, :]    = self.choiceRate[:, possExploits, 0].T
            probChoice[blockedExploits, :] = self.choiceRate[:, blockedExploits, 1].T
            probChoiceRate                 = np.prod(probChoice[chosenExploits, :], axis=0) * np.prod(1 - probChoice[notChosenExploits, :], axis=0)
            probAlert                      = 1 - (np.prod(1 - self.idsTPMatrix[a, chosenExploits, :], axis=0) * (1 - self.idsFPMatrix[a]))
            
            tpProb                         = 1 - np.prod(1 - self.idsTPMatrix[a, chosenExploits, :], axis=0)
            fpProb                         =     np.prod(1 - self.idsTPMatrix[a, chosenExploits, :], axis=0) * self.idsFPMatrix[a]
            pProb                          = 1 - (np.prod(1 - self.idsTPMatrix[a, chosenExploits, :], axis=0) * (1 - self.idsFPMatrix[a]))
            
            tpProbNumerator = ((fpProb * (1 - omega) + tpProb * omega) / pProb) * (probChoiceRate * probAlert)
            fpProbNumerator = ((tpProb * (1 - omega) + fpProb * omega) / pProb) * (probChoiceRate * probAlert)
            tp_ProbSum = ((fpProb * (1 - omega) + tpProb * omega) / pProb) * ((probChoiceRate * probAlert)) * sum_beta
            fp_ProbSum = ((tpProb * (1 - omega) + fpProb * omega) / pProb) * ((probChoiceRate * probAlert)) * sum_beta
            tpProbSum = np.divide(tpProbNumerator, denominator, out=np.zeros_like(tpProbNumerator), where=denominator!=0)
            fpProbSum = np.divide(fpProbNumerator, denominator, out=np.zeros_like(fpProbNumerator), where=denominator!=0)
            tp_ProbSum = np.divide(tp_ProbSum, denominator, out=np.zeros_like(tp_ProbSum), where=denominator!=0)
            fp_ProbSum = np.divide(fp_ProbSum, denominator, out=np.zeros_like(fp_ProbSum), where=denominator!=0)
        return tpProbSum, fpProbSum, tp_ProbSum, fp_ProbSum
    
    def maxEntropyPolicy_yk(self, alerts, trueAlerts, belief, omega, security_anlayst):
        p = "MaxEntropy"
        if len(alerts) != 0:
            entropies_wo_min = np.zeros([len(alerts)])
            investIdx = []
            investResult = []
            policy = np.zeros([2, len(alerts), self.feasibleStates.shape[0], self.numAttTypes])
            for aIdx, a in enumerate(alerts): 
                raw_idx = belief.sum(axis=1).argsort()[::-1]
                belief_cumsum = belief.sum(axis=1)[raw_idx].cumsum()
                flt_state_idx = raw_idx[belief_cumsum <= (1-self.state_threshold)]
                if len(flt_state_idx)==0:
                    flt_state_idx = raw_idx[:1]
                for iIdx, iBin in zip(flt_state_idx, self.iBins[flt_state_idx]):
                    e_s_i = self.graph.checkAvailExploit_bin(iBin)
                    e_s_i = e_s_i[e_s_i>=0]
                    tpProbSum, fpProbSum = self.tpAndfpProb_yk(iBin, iIdx, e_s_i, a, omega)
                    policy[0, aIdx, iIdx] = tpProbSum
                    policy[1, aIdx, iIdx] = fpProbSum
            policy = policy * belief
            trans_policy = policy.transpose(1,0,2,3)
            entropies_w_min = np.zeros([len(alerts), 2])
            for i, each_alert in enumerate(trans_policy):
                entropies_wo_min[i] = entr(each_alert).sum()/np.log(2)
                for j, true_false in enumerate(each_alert):
                    entropies_w_min[i,j] = entr(true_false).sum()/np.log(2)
            investIdx = alerts[np.min(np.abs(entropies_w_min), axis =1).argsort()[-self.ib[p][security_anlayst]:]]
            investIdx = alerts[entropies_wo_min.argsort()[-self.ib[p][security_anlayst]:]]
            investResult = trueAlerts[investIdx]                          
        else:
            investIdx = []
            investResult = []
        return (np.array(investIdx, dtype=np.int8).reshape(-1), np.array(investResult, dtype=np.bool).reshape(-1))
     
    def tpAndfpProb_yk(self, iBin, iIdx, e_s_i, a, omega):
        tpProbSum, fpProbSum = 0, 0
        denominator = 0
        for chosenExploits in self.state_e_powerset[iIdx]:
            chosenExploits                 = np.array(chosenExploits, dtype=np.int8)
            notChosenExploits              = np.setdiff1d(e_s_i, chosenExploits)

            probChoice                     = np.ones([self.numExploit, self.numAttTypes])
            blockedExploits                = self.graph.checkBlockedExploit(e_s_i)
            possExploits                   = np.setdiff1d(e_s_i, blockedExploits)
            
            probChoice[possExploits, :]    = self.choiceRate[:, possExploits, 0].T
            probChoice[blockedExploits, :] = self.choiceRate[:, blockedExploits, 1].T
            probChoiceRate                 = np.prod(probChoice[chosenExploits, :], axis=0) * np.prod(1 - probChoice[notChosenExploits, :], axis=0)

            probAlert                      = 1 - (np.prod(1 - self.idsTPMatrix[a, chosenExploits, :], axis=0) * (1 - self.idsFPMatrix[a]))
            denominator += probChoiceRate * probAlert
        
        for chosenExploits in powerSet(e_s_i):          
            chosenExploits                 = np.array(chosenExploits, dtype=np.int8)
            notChosenExploits              = np.setdiff1d(e_s_i, chosenExploits)

            probChoice                     = np.ones([self.numExploit, self.numAttTypes])
            blockedExploits                = self.graph.checkBlockedExploit(e_s_i)
            possExploits                   = np.setdiff1d(e_s_i, blockedExploits)
            
            probChoice[possExploits, :]    = self.choiceRate[:, possExploits, 0].T
            probChoice[blockedExploits, :] = self.choiceRate[:, blockedExploits, 1].T
            probChoiceRate                 = np.prod(probChoice[chosenExploits, :], axis=0) * np.prod(1 - probChoice[notChosenExploits, :], axis=0)
            probAlert                      = 1 - (np.prod(1 - self.idsTPMatrix[a, chosenExploits, :], axis=0) * (1 - self.idsFPMatrix[a]))
            
            tpProb                         = 1 - np.prod(1 - self.idsTPMatrix[a, chosenExploits, :], axis=0)
            fpProb                         =     np.prod(1 - self.idsTPMatrix[a, chosenExploits, :], axis=0) * self.idsFPMatrix[a]
            pProb                          = 1 - (np.prod(1 - self.idsTPMatrix[a, chosenExploits, :], axis=0) * (1 - self.idsFPMatrix[a]))
            
            tpProbNumerator = ((fpProb * (1 - omega) + tpProb * omega) / pProb) * (probChoiceRate * probAlert)
            fpProbNumerator = ((tpProb * (1 - omega) + fpProb * omega) / pProb) * (probChoiceRate * probAlert)
            tpProbSum = np.divide(tpProbNumerator, denominator, out=np.zeros_like(tpProbNumerator), where=denominator!=0)
            fpProbSum = np.divide(fpProbNumerator, denominator, out=np.zeros_like(fpProbNumerator), where=denominator!=0)
        return tpProbSum, fpProbSum

    def bayesFactorPolicy(self, alerts, trueAlerts, belief, omega, security_anlayst):
        p = "Bayes"
        investIdx = []
        investResult = []
        if len(alerts) != 0:
            policy = np.zeros([len(alerts), self.feasibleStates.shape[0], self.numAttTypes])
            for aIdx, a in enumerate(alerts):
                minus_a_positive = np.setdiff1d(alerts, a)
                minus_a_negative = np.setdiff1d(np.arange(self.numAlerts, dtype=np.int8), alerts)
                raw_idx = belief.sum(axis=1).argsort()[::-1]
                belief_cumsum = belief.sum(axis=1)[raw_idx].cumsum()
                flt_state_idx = raw_idx[belief_cumsum <= (1-self.state_threshold)]
                if len(flt_state_idx)==0:
                    flt_state_idx = raw_idx[:1]
                for iIdx, iBin in zip(flt_state_idx, self.iBins[flt_state_idx]):
                    e_s_i = self.graph.checkAvailExploit_bin(iBin)
                    e_s_i = e_s_i[e_s_i>=0]
                    probAlertAndExploits_zeta0, probAlertAndExploits = self.cal_alert_prob(e_s_i, minus_a_positive, minus_a_negative, a, omega)
                    policy[aIdx, iIdx] = probAlertAndExploits_zeta0 / probAlertAndExploits          
            numerator = (policy*belief).transpose(1,0,2)
            denominator = self.idsFPMatrix[alerts]
            policy = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0) - 1
            policy = policy.transpose(1,0,2)
            each_alert = np.sum(np.sum(policy**2, axis=1),axis=1)
            investIdx = alerts[each_alert.argsort()[:self.ib[p][security_anlayst]]]
            investResult = trueAlerts[investIdx]
        else:
            investIdx = []
            investResult = []
        return (np.array(investIdx, dtype=np.int8).reshape(-1), np.array(investResult, dtype=np.bool).reshape(-1))

    def cal_alert_prob(self, e_s_i, minus_a_positive, minus_a_negative, a, omega):
        probAlertAndExploits_zeta0 = np.zeros(self.numAttTypes)
        probAlertAndExploits = np.zeros(self.numAttTypes)
        for chosenExploits in powerSet(e_s_i):
            probAlert                         = np.ones([self.numAlerts, self.numAttTypes])
            probAlert_zeta0                   = np.ones([self.numAlerts, self.numAttTypes])
            chosenExploits                    = np.array(chosenExploits, dtype=np.int8)
            notChosenExploits                 = np.setdiff1d(e_s_i, chosenExploits)

            probChoice                        = np.ones([self.numExploit, self.numAttTypes])
            blockedExploits                   = self.graph.checkBlockedExploit(e_s_i)
            possExploits                      = np.setdiff1d(e_s_i, blockedExploits)
            
            probChoice[possExploits, :]       = self.choiceRate[:, possExploits, 0].T
            probChoice[blockedExploits, :]    = self.choiceRate[:, blockedExploits, 1].T
            probChoiceRate                    = np.prod(probChoice[chosenExploits, :], axis=0) * np.prod(1 - probChoice[notChosenExploits, :], axis=0)

            probAlert[minus_a_positive]       = 1 - (np.prod(1 - self.idsTPMatrix[minus_a_positive, :, :][:, chosenExploits, :], axis=1) * (1 - self.idsFPMatrix[minus_a_positive]))
            probAlert[minus_a_negative]       =      np.prod(1 - self.idsTPMatrix[minus_a_negative, :, :][:, chosenExploits, :], axis=1) * (1 - self.idsFPMatrix[minus_a_negative])
            probAlertRate                     = np.prod(probAlert, axis = 0) # Probability of the current alert by chosen exploits
            
            probAlert_zeta0[a]                = (1 - np.prod(1 - self.idsTPMatrix[a][chosenExploits], axis=0)) * omega #+ np.prod(1 - self.idsTPMatrix[a][chosenExploits], axis=0) * (1 - self.idsFPMatrix[a]) * (1 - omega)
            probAlert_zeta0[minus_a_positive] = 1 - (np.prod(1 - self.idsTPMatrix[minus_a_positive, :, :][:, chosenExploits, :], axis=1) * (1 - self.idsFPMatrix[minus_a_positive]))
            probAlert_zeta0[minus_a_negative] =      np.prod(1 - self.idsTPMatrix[minus_a_negative, :, :][:, chosenExploits, :], axis=1) * (1 - self.idsFPMatrix[minus_a_negative])
            probAlertRate_zeta0               = np.prod(probAlert_zeta0, axis = 0)
            
            probAlertAndExploits_zeta0        += probAlertRate_zeta0 * probChoiceRate 
            probAlertAndExploits              += probAlertRate * probChoiceRate
        return probAlertAndExploits_zeta0, probAlertAndExploits

    def rlPolicy(self, alerts, trueAlerts, belief, name):
        y_t_vector = np.zeros(self.numAlerts+1, dtype=float)
        if len(alerts) == 0:
            y_t_vector[-1] =  1
        else:
            y_t_vector[alerts] =  1
        observation = np.concatenate([belief.reshape(-1), y_t_vector.reshape(-1)], axis= 0)
        logit = self.RL_policy.forward({'obs':torch.Tensor(observation)})['action_dist_inputs']
        policy = torch.softmax(logit,dim=0)
        alerts_tensor = torch.tensor(alerts)
        if len(alerts) == 0 :
            investIdx = self.numAlerts
        else:
            investIdx = alerts_tensor[policy.reshape(-1)[alerts_tensor].argmax()].numpy()
        if investIdx in alerts:
            investResult = trueAlerts[investIdx]
        else:
            investIdx = []
            investResult = []    
        return (np.array(investIdx, dtype=np.int8).reshape(-1), np.array(investResult, dtype=np.bool).reshape(-1))

    def build_p_iiʼl(self):
        p_iiʼl = np.zeros([self.feasibleStates.shape[0], self.feasibleStates.shape[0], self.numAttTypes])
        for iIdx, iBin in enumerate(self.iBins):
            availExploits = self.graph.checkAvailExploit_bin(iBin)
            availExploits = availExploits[availExploits>=0]
            for exploits in self.state_e_powerset[iIdx]:
                v1= np.array(exploits, dtype=int)
                v0= np.array(np.setdiff1d(availExploits, v1), dtype=int)
                
                iʼBin = (iBin + np.sum(self.postCondition[v1], axis=0)) > 0
                iʼ = iʼBin.dot(2**np.arange(iʼBin.size))
                
                alpha_te = np.ones([self.numAttTypes, self.numExploit])
                beta_te = np.ones([self.numAttTypes, self.numExploit])
                
                alpha_te[:, availExploits] = self.choiceRate[:, availExploits, 0]
                beta_te[:, availExploits] = self.successRate[:, availExploits]
                
                iʼProb = alpha_te * beta_te
                iʼProb[:, v0] = 1 - iʼProb[:, v0] 

                p_iiʼ = np.prod(iʼProb, axis = 1)
                
                iʼIdx = self.idxFeasibleStates[iʼ] 
                p_iiʼl[iʼIdx, iIdx, :] += p_iiʼ
        return p_iiʼl

    def cal_p_iiʼl(self, y_t, belief):
        exploits_combi = 2 ** max([len(v) for k,v in self.feasibleExploits.items()])
        alerts_n = np.array(y_t, dtype=int)
        noAlerts = np.setdiff1d(np.arange(self.numAlerts), y_t)
        p_eiiʼl = np.zeros([exploits_combi, self.feasibleStates.shape[0], self.feasibleStates.shape[0], self.numAttTypes])
        raw_idx = belief.sum(axis=1).argsort()[::-1]
        belief_cumsum = belief.sum(axis=1)[raw_idx].cumsum()
        flt_state_idx = raw_idx[belief_cumsum <= (1-self.state_threshold)]
        if len(flt_state_idx)<= 0:
            flt_state_idx = raw_idx[:1]
        for iIdx, iBin in zip(flt_state_idx, self.iBins[flt_state_idx]):
            availExploits = self.graph.checkAvailExploit_bin(iBin)
            availExploits = availExploits[availExploits>=0]
            blockedExploits = self.graph.checkBlockedExploit(availExploits)
            possExploits = np.setdiff1d(availExploits, blockedExploits)
            choiceRate = np.ones([self.numExploit, self.numAttTypes])
            choiceRate[possExploits, :] = self.choiceRate[:, possExploits, 0].T
            choiceRate[blockedExploits, :] = self.choiceRate[:, blockedExploits, 1].T
            for eIdx, E_t in enumerate(self.state_e_powerset[iIdx]):
                sum_beta = np.zeros([len(self.feasibleStates), self.numAttTypes])
                not_E_t = np.setdiff1d(availExploits, E_t)
                alpha = np.prod(choiceRate[E_t, :], axis=0) * np.prod(1 - choiceRate[not_E_t, :], axis=0)
                probAlert_raw = np.ones([self.numAlerts,self.numAttTypes])
                probAlert_raw[alerts_n, :] = 1 - np.prod(1 - self.idsTPMatrix[alerts_n, :, :][:, E_t, :], axis=1) * (1 - self.idsFPMatrix[alerts_n])
                probAlert_raw[noAlerts, :] = np.prod(1 - self.idsTPMatrix[noAlerts, :, :][:, E_t, :], axis=1) * (1 - self.idsFPMatrix[noAlerts])
                probCurrAlert_raw = np.prod(probAlert_raw, axis = 0)
                for v1 in powerSet(E_t):
                    v1= np.array(v1, dtype=int)
                    v0= np.array(np.setdiff1d(E_t, v1), dtype=int)

                    beta = np.ones([self.numAttTypes, self.numExploit])
                    beta[:, v1] = self.successRate[:, v1]
                    beta[:, v0] = 1 - self.successRate[:, v0] 

                    binNextPossStates = (iBin + np.sum(self.postCondition[v1], axis=0)) > 0
                    j = binNextPossStates.dot(2**np.arange(binNextPossStates.size))
                    jIdx = self.idxFeasibleStates[j] 
                    sum_beta[jIdx, :] += np.prod(beta, axis=1)
                p_eiiʼl[eIdx,:,iIdx,:] += probCurrAlert_raw * alpha * sum_beta
        margin_p_iiʼl = p_eiiʼl.sum(axis=0)
        # margin_p_iiʼl = np.divide(p_iiʼl, p_iiʼl.sum(axis=0), out=np.zeros_like(p_iiʼl), where=p_iiʼl.sum(axis=0)!=0)
        if np.isnan(margin_p_iiʼl).any():
            print('stop')
        return margin_p_iiʼl

    def r_iiʼnkl(self, alerts_n, p_ijl, investigation, belief, omega=[1]):
        filteredAlert, investResult = investigation
        exploits_combi = 2 ** max([len(v) for k,v in self.feasibleExploits.items()])
        rawAlerts = np.setdiff1d(alerts_n, filteredAlert)
        noAlerts = np.setdiff1d(np.arange(self.numAlerts), alerts_n)
        r_eijkl = np.zeros([exploits_combi, p_ijl.shape[0], p_ijl.shape[0], self.numAttTypes])
        raw_idx = belief.sum(axis=1).argsort()[::-1]
        belief_cumsum = belief.sum(axis=1)[raw_idx].cumsum()            
        flt_state_idx = raw_idx[belief_cumsum <= (1  - self.state_threshold)]
        if len(flt_state_idx)<= 0:
            flt_state_idx = raw_idx[:1]
        for iIdx, iBin in zip(flt_state_idx, self.iBins[flt_state_idx]):
            investigations = np.zeros([exploits_combi, self.numAttTypes])
            availExploits = self.graph.checkAvailExploit_bin(iBin)
            availExploits = availExploits[availExploits>=0]
            blockedExploits = self.graph.checkBlockedExploit(availExploits)
            possExploits = np.setdiff1d(availExploits, blockedExploits)
            choiceRate = np.ones([self.numExploit, self.numAttTypes])
            choiceRate[possExploits, :] = self.choiceRate[:, possExploits, 0].T
            choiceRate[blockedExploits, :] = self.choiceRate[:, blockedExploits, 1].T
            for eIdx, E_t in enumerate(self.state_e_powerset[iIdx]):
                not_E_t = np.setdiff1d(availExploits, E_t)
                alpha = np.prod(choiceRate[E_t, :], axis=0) * np.prod(1 - choiceRate[not_E_t, :], axis=0)     
                probAlert_filtered = np.ones([self.numAlerts,self.numAttTypes])
                probAlert_filtered[rawAlerts, :] = 1
                probAlert_filtered[noAlerts, :] = 1
                # for j in filteredAlert[investResult]:
                #     numerator = ((1 - np.prod(1 - self.idsTPMatrix[j][E_t,:], axis=0)) * omega + self.idsFPMatrix[j] * np.prod(1 - self.idsTPMatrix[j][E_t,:], axis=0) * (1 -omega) ) 
                #     denominator = (1 - np.prod(1 - self.idsTPMatrix[j][E_t, :], axis=0) * (1 - self.idsFPMatrix[j]))
                #     probAlert_filtered[j] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0) 
                # for j in filteredAlert[~investResult]:
                #     numerator = ((1 - np.prod(1 - self.idsTPMatrix[j][E_t,:], axis=0)) * (1 - omega) + self.idsFPMatrix[j] * np.prod(1 - self.idsTPMatrix[j][E_t,:], axis=0) * omega ) 
                #     denominator = (1 - np.prod(1 - self.idsTPMatrix[j][E_t, :], axis=0) * (1 - self.idsFPMatrix[j]))
                #     probAlert_filtered[j] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0) 
                for j, o in zip(filteredAlert[investResult], omega[investResult]):
                    numerator = ((1 - np.prod(1 - self.idsTPMatrix[j][E_t,:], axis=0)) * o + self.idsFPMatrix[j] * np.prod(1 - self.idsTPMatrix[j][E_t,:], axis=0) * (1 - o) ) 
                    denominator = (1 - np.prod(1 - self.idsTPMatrix[j][E_t, :], axis=0) * (1 - self.idsFPMatrix[j]))
                    probAlert_filtered[j] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0) 
                for j, o in zip(filteredAlert[~investResult], omega[~investResult]):
                    numerator = ((1 - np.prod(1 - self.idsTPMatrix[j][E_t,:], axis=0)) * (1 - o) + self.idsFPMatrix[j] * np.prod(1 - self.idsTPMatrix[j][E_t,:], axis=0) * o ) 
                    denominator = (1 - np.prod(1 - self.idsTPMatrix[j][E_t, :], axis=0) * (1 - self.idsFPMatrix[j]))
                    probAlert_filtered[j] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0) 
                investigations[eIdx] = np.prod(probAlert_filtered, axis = 0)
                sum_beta = np.zeros([p_ijl.shape[0], self.numAttTypes])
                probAlert_raw = np.ones([self.numAlerts,self.numAttTypes])
                probAlert_raw[alerts_n, :] = 1 - np.prod(1 - self.idsTPMatrix[alerts_n, :, :][:, E_t, :], axis=1) * (1 - self.idsFPMatrix[alerts_n])
                probAlert_raw[noAlerts, :] = np.prod(1 - self.idsTPMatrix[noAlerts, :, :][:, E_t, :], axis=1) * (1 - self.idsFPMatrix[noAlerts])
                probCurrAlert_raw = np.prod(probAlert_raw, axis = 0)
                for v1 in powerSet(E_t):
                    v1= np.array(v1, dtype=int)
                    v0= np.array(np.setdiff1d(E_t, v1), dtype=int)

                    beta = np.ones([self.numAttTypes, self.numExploit])
                    beta[:, v1] = self.successRate[:, v1]
                    beta[:, v0] = 1 - self.successRate[:, v0] 

                    binNextPossStates = (iBin + np.sum(self.postCondition[v1], axis=0)) > 0
                    j = binNextPossStates.dot(2**np.arange(binNextPossStates.size))
                    jIdx = self.idxFeasibleStates[j] 
                    sum_beta[jIdx, :] += np.prod(beta, axis=1)
                r_eijkl[eIdx, :, iIdx, :] = probCurrAlert_raw * sum_beta * alpha
            r_eijkl[:,:, iIdx, :] = np.divide(r_eijkl[:,:, iIdx, :], p_ijl[:,iIdx,:], out=np.zeros_like(r_eijkl[:,:, iIdx, :]), where=p_ijl[:,iIdx,:]!=0) 
            margin_r_eijkl = np.divide(r_eijkl[:, :, iIdx, :], r_eijkl.sum(axis=0)[:, iIdx, :], \
                                                out=np.zeros_like(r_eijkl[:, :, iIdx, :]), where=r_eijkl.sum(axis=0)[:, iIdx, :]!=0) 
            r_eijkl[:, :, iIdx, :] = (investigations * np.transpose(margin_r_eijkl, (1,0,2))).transpose(1,0,2)
        output_r_ijkl= r_eijkl.sum(axis=0)
        if np.isnan(output_r_ijkl).any():
            print('r_ijkl stop')
        return output_r_ijkl

    def cal_belief(self, belief, p, r, investigation = None):
        if (r==0).all():
            r = self.r_iiʼnkl_no
            p_iʼm        = np.sum(np.dot(belief * p, self.q), axis = 1)
            r_iʼk        = np.sum(belief * r, axis = 1)
            rawBelief    = p_iʼm * r_iʼk
            newBelief    = rawBelief / np.sum(rawBelief)
            # newBelief[newBelief<self.epsilon] = self.epsilon
            self.cnt_non_zero += 1
            # print("cal_belief")
        else:
            p_iʼm        = np.sum(np.dot(belief * p, self.q), axis = 1)
            r_iʼk        = np.sum(belief * r, axis = 1)
            rawBelief    = p_iʼm * r_iʼk
            newBelief    = rawBelief / np.sum(rawBelief)
            # newBelief[newBelief<self.epsilon] = self.epsilon
            self.cnt_zero += 1
        newBelief = newBelief/ newBelief.sum() #To avoid errors by using EPS instead of zeros.
        return newBelief

    def skip(self):
        result = {}
        for policy, belief_dict in self.beliefDict.items():
            for omega, belief in belief_dict.items():
                if omega =="untrained":
                    name = omega
                    omega = 1
                else:
                    name = omega
                self.beliefDict[policy][name] = self.beliefDict[policy][name]
                result[f'{policy}_{name}'] = (self.beliefDict[policy][name],  (np.array([], dtype = int ), np.array([], dtype=np.bool)))
        return result
                                                
    def updateBelief(self, y_t, y_t_true, attState, alerts_n, trueAlerts, E_t):
        result = {}
        self.E_t_succeed = E_t
        self.step += 1
        for omega in self.w_list:
            if omega[0] =="untrained":
                name = omega[0]
                omega = 1
            else:
                name = omega[0]
            rand_error_prob = self.mistake_omega_random_state[omega[0]].rand(*[len(omega), y_t.shape[0]])
            for policy in self.policies:
                start = time.time()
                p_iiʼl = self.cal_p_iiʼl(y_t, self.beliefDict[policy][omega[0]])
                investigation_lst = []
                for security_anlayst, o in enumerate(omega):
                    y_t_hat = np.zeros(self.numAlerts, dtype= bool)
                    y_t_hat[y_t_true] = True
                    mistake = y_t[rand_error_prob[security_anlayst] > o]
                    y_t_hat[mistake] = ~y_t_hat[mistake]
                    alert_k_result = (np.array(y_t, dtype = int ), np.array(y_t_hat[y_t], dtype=np.bool))
                    if "No Investigation" == policy:
                        investigation = (np.array([], dtype=np.int8).reshape(-1), np.array([], dtype=np.bool).reshape(-1))
                    elif "Bayes" == policy:
                        investigation = self.bayesFactorPolicy(y_t, y_t_hat, self.beliefDict[policy][name], o, security_anlayst)
                    elif "MaxEntropy" == policy:
                        investigation = self.maxEntropyPolicy(y_t, y_t_hat, self.beliefDict[policy][name], o, security_anlayst)
                    elif "MinFP" == policy:
                        investigation = self.lowFPPolicy(y_t, y_t_hat, self.beliefDict[policy][name], o, security_anlayst)
                    elif "Random" == policy:
                        investigation = self.randomPolicy(y_t, y_t_hat, o, security_anlayst)
                    elif "RL" == policy:
                        investigation = self.rlPolicy(y_t, y_t_hat, self.beliefDict[policy][name], name) 
                    elif "All" == policy:
                        investigation = alert_k_result
                    investigation_lst.append(investigation)
                    
                ##### Distinct investigation error #####
                final_investigation_decision = []
                final_investigation_outcomes = []
                omega_lst = []
                for inv in investigation_lst:
                    for human, budget in enumerate(self.ib[policy]):
                        for jdx, alert_to_investigate in enumerate(inv[0]):
                            if alert_to_investigate not in final_investigation_decision and len(final_investigation_decision)<budget:
                                final_investigation_decision.append(alert_to_investigate)
                                final_investigation_outcomes.append(inv[1][jdx])
                                omega_lst.append(omega[human])
                final_investigation_result = (np.array(final_investigation_decision, dtype=np.int8), np.array(final_investigation_outcomes, dtype=np.bool).reshape(-1))
                omega_lst = np.array(omega_lst)
                ##### Distinct investigation error #####

                r_iiʼnkl   = self.r_iiʼnkl(y_t, p_iiʼl, final_investigation_result, self.beliefDict[policy][name], omega=omega_lst)
                if policy == "No Investigation":
                    self.r_iiʼnkl_no = r_iiʼnkl
                new_belief = self.cal_belief(self.beliefDict[policy][name], p_iiʼl, r_iiʼnkl, investigation)
                if np.isnan(new_belief).any(): #Avoid division by 0 by using epsilon again.
                    self.beliefDict[policy][name][self.beliefDict[policy][name]<EPS] = EPS
                    new_belief = self.cal_belief(self.beliefDict[policy][name], p_iiʼl, r_iiʼnkl, investigation)
                if np.isnan(new_belief).any(): #Avoid division by 0 by using epsilon again.
                    temp_omega = np.ones_like(omega_lst)-EPS
                    r_iiʼnkl   = self.r_iiʼnkl(y_t, p_iiʼl, final_investigation_result, self.beliefDict[policy][name], omega=temp_omega)
                    new_belief = self.cal_belief(self.beliefDict[policy][name], p_iiʼl, r_iiʼnkl, investigation)
                self.beliefDict[policy][name] = new_belief
                result[f'{policy}_{name}'] = (self.beliefDict[policy][name], investigation)
                self.time_dict[policy][name] += time.time() - start
        return result
    
    
    def createInvResult(self, inv, policy):
        ##### Distinct investigation error #####
        final_investigation_decision = []
        final_investigation_outcomes = []
        omega_lst = []
        for human, budget in enumerate(self.ib[policy]):
            for jdx, alert_to_investigate in enumerate(inv[0]):
                if alert_to_investigate not in final_investigation_decision and len(final_investigation_decision)<budget:
                    final_investigation_decision.append(alert_to_investigate)
                    final_investigation_outcomes.append(inv[1][jdx])
                    omega_lst.append(self.w_list[0])
        final_investigation_result = (np.array(final_investigation_decision, dtype=np.int8), np.array(final_investigation_outcomes, dtype=np.bool).reshape(-1))
        return final_investigation_result, omega_lst
    
    def oneUpdate(self, i, action, y_t, y_t_true):
        confidence = self.w_list[0][0]
        y_t_hat = np.zeros(self.numAlerts, dtype= bool)
        y_t_hat[y_t_true] = True
        mistake = y_t[np.random.rand(*y_t.shape) > confidence]
        y_t_hat[mistake] = ~y_t_hat[mistake]
        alert_k_result = y_t_hat[y_t]
        
        ## RL no investigation
        p_iiʼl = self.cal_p_iiʼl(y_t, self.beliefRL)
        investigationRL = (np.array([], dtype=np.int8), None)
        No_result, omega_lst = self.createInvResult(investigationRL, 'No Investigation')
        r_iiʼnkl   = self.r_iiʼnkl(y_t, p_iiʼl, No_result, self.beliefRL, omega=np.array(omega_lst))
        self.r_iiʼnkl_no = r_iiʼnkl
        beliefRL_no_inv = self.cal_belief(self.beliefRL, p_iiʼl, r_iiʼnkl)

        single_inv_belief_lst = []
        for a in y_t:
            investigation_temp = [np.array([a], dtype=np.int8), [y_t_hat[a]]]
            ##### Distinct investigation error #####
            final_investigation_result, omega_lst = self.createInvResult(investigation_temp, 'RL')
            omega_lst = np.array(omega_lst)
            ##### Distinct investigation error #####
            r_iiʼnkl   = self.r_iiʼnkl(y_t, p_iiʼl, final_investigation_result, self.beliefRL, omega=omega_lst) 
            new_belief = self.cal_belief(self.beliefRL, p_iiʼl, r_iiʼnkl, investigation_temp)
            single_inv_belief_lst.append(new_belief)


        ## All investigation
        # p_iiʼl_All = self.cal_p_iiʼl(y_t, self.beliefRL)
        investigationAll = [np.array(y_t, dtype=np.int8), y_t_hat[y_t]]
        All_result, omega_lst = self.createInvResult(investigationAll, 'All')
        r_iiʼnkl_All   = self.r_iiʼnkl(y_t, p_iiʼl, All_result, self.beliefRL, omega=np.array(omega_lst))
        beliefRL_All_inv = self.cal_belief(self.beliefRL, p_iiʼl, r_iiʼnkl_All, investigationAll)
        ## All investigation

        if action in y_t:

            investigationRL = [np.array([action], dtype=np.int8), [y_t_hat[action]]]
            ##### Distinct investigation error #####
            final_investigation_result, omega_lst = self.createInvResult(investigationRL, 'RL')
            omega_lst = np.array(omega_lst)
            ##### Distinct investigation error #####
            r_iiʼnkl   = self.r_iiʼnkl(y_t, p_iiʼl, final_investigation_result, self.beliefRL, omega=omega_lst) 
            new_belief = self.cal_belief(self.beliefRL, p_iiʼl, r_iiʼnkl, investigationRL)
            self.beliefRL = new_belief
        else:
            self.beliefRL = beliefRL_no_inv
        return self.beliefRL, investigationRL, beliefRL_no_inv, beliefRL_All_inv, single_inv_belief_lst

if __name__ == "__main__":
    attackerType = "High"
    # IDS
    idsRatePath = "config/IDS/idsRate.json"
    # Attacker 
    choicePath = "config/attacker/attackChoice.json"
    successPath = "config/attacker/attackSuccess.json"
    # Defender
    effectPath = "config/environment/defenseEffect.json"
    # Graph
    graphPath = "config/environment/dependGraph.json"    

    testState = 0
    testDefenses = 0
    alerts = np.array([0,1,2,3,4,5,6,7])

    ids = IDS(attackerType, idsRatePath)
    graph = Graph(attackerType, graphPath, effectPath, testState, testDefenses)
    defender = Defender(ids, graph, choicePath, successPath, testDefenses)
    
    defender.updateBelief(alerts)