# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import numpy as np

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #if state not in self.values: add it with value 0
                
        for i in range(self.iterations):
            values = self.values.copy()
            allStates = self.mdp.getStates()
            for s in allStates:
                if not self.mdp.isTerminal(s):
                    maxValue = float('-inf')
                    actions = self.mdp.getPossibleActions(s)
                    for a in actions:
                        transitions = self.mdp.getTransitionStatesAndProbs(s, a)
                        oneAction = 0
                        for t in transitions:
                            nextState, prob = t
                            reward = self.mdp.getReward(s, a, nextState)
                            oneTransition = prob * (reward + self.discount * self.values[nextState])
                            oneAction = oneAction + oneTransition
                        if oneAction > maxValue:
                            maxValue = oneAction
                    values[s] = maxValue
            self.values = values 

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        #print('Transitions: ' + str(transitions))
        for t in transitions:
            nextState, prob = t
            reward = self.mdp.getReward(state, action, nextState)
            #print('Reward: ' + str(reward))
            oneTransition = prob * (reward + self.discount * self.values[nextState])
            qValue = qValue + oneTransition
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        optimalAction = None
        maxValue = float('-inf')
        for a in actions:
            qValue = self.computeQValueFromValues(state, a)
            if qValue > maxValue:
                maxValue = qValue
                optimalAction = a
        return optimalAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        allStates = self.mdp.getStates()
        for i in range(self.iterations):
            values = self.values.copy()
            counter = i % len(allStates)
            s = allStates[counter]
            if not self.mdp.isTerminal(s):
                maxValue = float('-inf')
                actions = self.mdp.getPossibleActions(s)
                for a in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(s, a)
                    oneAction = 0
                    for t in transitions:
                        nextState, prob = t
                        reward = self.mdp.getReward(s, a, nextState)
                        oneTransition = prob * (reward + self.discount * self.values[nextState])
                        oneAction = oneAction + oneTransition
                    if oneAction > maxValue:
                        maxValue = oneAction
                values[s] = maxValue
            self.values = values 

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #compute predecessors of all states
        predecessors = {}
        allStates = self.mdp.getStates()
        for s in allStates:
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                for a in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(s, a)
                    for t in transitions:
                        nextState, prob = t
                        if prob > 0:
                            if nextState not in predecessors:
                                predecessors[nextState] = set()
                            predecessors[nextState].add(s)
        
        #initialize empty priotiy queue
        pqueue = util.PriorityQueue()

        #compute differences
        for s in allStates:
            if not self.mdp.isTerminal(s):
                currVal = self.values[s]
                maxqVal = float('-inf')
                actions = self.mdp.getPossibleActions(s)
                for a in actions:
                    q = self.computeQValueFromValues(s, a)
                    if q > maxqVal:
                        maxqVal = q
                diff = maxqVal - currVal
                pqueue.update(s, -abs(diff))
        
        #update values
        for i in range(self.iterations):
            if not pqueue.isEmpty():
                state = pqueue.pop()
                if not self.mdp.isTerminal(state):
                    #self.values[state] = differences[state]
                    maxqVal = float('-inf')
                    actions = self.mdp.getPossibleActions(state)
                    for a in actions:
                        q = self.computeQValueFromValues(state, a)
                        if q > maxqVal:
                            maxqVal = q
                    self.values[state] = maxqVal

            preds = predecessors[state]
            for p in preds:
                if not self.mdp.isTerminal(p):
                    currPVal = self.values[p]
                    maxqVal = float('-inf')
                    actions = self.mdp.getPossibleActions(p)
                    for a in actions:
                        q = self.computeQValueFromValues(p, a)
                        if q > maxqVal:
                            maxqVal = q
                    diff = abs(currPVal - maxqVal)
                    if diff > abs(self.theta):
                        pqueue.update(p, -diff)