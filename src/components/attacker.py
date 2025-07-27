
import json
import numpy as np
try:
    from src.util.auxiliaries import checkAttackerType, convKeyInt, convertDec2Bin
    from src.components.ids import IDS
    from src.components.graph import Graph
except:
    import os, sys
    p = os.path.abspath('.')
    sys.path.insert(1, p)
    from src.components.ids import IDS
    from src.components.graph import Graph
    from src.util.auxiliaries import checkAttackerType, convKeyInt, convertDec2Bin

class Attacker:
    def __init__(self, ids : IDS, graph : Graph, attackerType, choicePath, successPath, seed, testState = 0):
        self.ids = ids
        self.graph = graph
        self.numExploit = self.graph.numExploit
        self.attackerType = attackerType
        with open(choicePath, 'r') as f:
            choiceRate = json.load(f)
        choiceRate = checkAttackerType(attackerType, choiceRate)
        choiceRate = convKeyInt(choiceRate)
        self.choiceRate = np.zeros([self.numExploit,2])
        for k,v in choiceRate.items():
            self.choiceRate[k-1] = v
        with open(successPath, 'r') as f:
            successRate = json.load(f)
        successRate = checkAttackerType(attackerType, successRate)
        successRate = convKeyInt(successRate)
        self.successRate = np.array([successRate[k] for k in sorted(successRate.keys())])
        self.state = testState
        self.random_state = np.random.RandomState(seed+1)

    def chooseExploits(self):
        availExploit = self.graph.checkAvailExploit(self.state)
        availExploit = availExploit[availExploit>=0]
        blockedExploit = np.intersect1d(availExploit, self.graph.defensedExploits)
        possibleExploit = np.setdiff1d(availExploit, blockedExploit)
        
        probChoice = np.zeros([self.choiceRate.shape[0]])
        probChoice[possibleExploit] = self.choiceRate[possibleExploit, 0]
        probChoice[blockedExploit] = self.choiceRate[blockedExploit, 1]
        
        E_t = np.arange(self.numExploit)[self.random_state.rand(*probChoice.shape) < probChoice] 
        E_t_blocked = np.intersect1d(E_t, blockedExploit)
        E_t_possible = np.setdiff1d(E_t, E_t_blocked)

        return E_t, E_t_possible, E_t_blocked
    
    def performExploits(self, E_t_possible):
        probSuccess = []
        probSuccess = self.successRate[E_t_possible]
        random_value = self.random_state.rand(*E_t_possible.shape) 
        E_t_succeed = E_t_possible[random_value < probSuccess]
        self.state = self.graph.updateState(E_t_succeed)
        return self.state, E_t_succeed

if __name__ == "__main__":
    attackerType = "Inter"
    # IDS
    idsRatePath = "IDS/idsRate.json"
    # Attacker 
    choicePath = "attacker/attackChoice.json"
    successPath = "attacker/attackSuccess.json"
    # Defender
    detectPath = "defender/defendDetect.json"
    effectPath = "environment/defenseEffect.json"
    # Graph
    # effectPath = "environment/defenseEffect.json"
    graphPath = "environment/dependGraph.json"

    testState = 15
    testDefenses = 2

    ids = IDS(attackerType, idsRatePath)
    graph = Graph(attackerType, graphPath, effectPath, testState, testDefenses)
    attacker = Attacker(ids, graph, attackerType, choicePath, successPath, testState)