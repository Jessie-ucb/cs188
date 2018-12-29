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
        states = self.mdp.getStates()
        " initialize self.values = 0"
        for state in states:
            self.values[state] = 0
        " value iteration"
        for k in range(self.iterations):
            valuesCopy = self.values.copy()  
            for state in states:
                #print("value iteration for state:", state)
                maxValue = -float("inf")
                for action in self.mdp.getPossibleActions(state):
                    QValue = self.computeQValueFromValues(state, action)
                    #print("QValue:", QValue)
                    if QValue > maxValue: maxValue = QValue
                if self.mdp.isTerminal(state): maxValue = 0
                #print("maxValue:", maxValue)
                valuesCopy[state] = maxValue  # update valuesCopy one single weight by another
            #print("Current round of Value Iteration's result:", valuesCopy)
            self.values = valuesCopy        # update the whole vector
      

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
        QValue = 0
        for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
          QValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))
        return QValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #print("State:", state)
        if self.mdp.isTerminal(state): 
          #print("action: none")
          return None
        #print("Value:", self.getValue(state))
        for action in self.mdp.getPossibleActions(state):
          #print("QValue:", self.computeQValueFromValues(state, action))
          if abs(self.computeQValueFromValues(state, action) - self.getValue(state)) < 1e-5:  # vs. strict equal ==
            #print("Matching action found! Values are: ", self.values)
            return action
        #print("No matching action found! Return none. Values are: ", self.values)
        return None

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
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for state in states:
            self.values[state] = 0
        #print("Starting values of states:", self.values)
        #print("number of iterations:", self.iterations)

        k = 0
        while k < self.iterations:
            for state in states:
                #print("value iteration for state:", state)
                maxValue = -float("inf")
                for action in self.mdp.getPossibleActions(state):
                    QValue = self.computeQValueFromValues(state, action)
                    #print("QValue:", QValue)
                    if QValue > maxValue: maxValue = QValue
                if self.mdp.isTerminal(state): maxValue = 0
                #print("maxValue:", maxValue)
                self.values[state] = maxValue
                #print("values of states:", self.values)
                k += 1
                if k >= self.iterations: break
            

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
        def getPredecessors(state):
            predecessors = set()
            for cand in self.mdp.getStates(): # candidatePredecessors:
                for action in self.mdp.getPossibleActions(cand):
                    for nextstate, prob in self.mdp.getTransitionStatesAndProbs(cand, action):
                        if nextstate == state and prob > 0:
                            predecessors.add(cand)
            return predecessors
        
        def calcDiff(state):
            maxQValue = -float("inf")
            for action in self.mdp.getPossibleActions(state):
                QValue = self.computeQValueFromValues(state, action)
                if QValue > maxQValue: maxQValue = QValue
            return abs(maxQValue - self.values[state])

        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0
        
        candidateStates = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                candidateStates.update(state, -calcDiff(state))

        #print("Starting values of states:", self.values)
        #print("number of iterations:", self.iterations)
        k = 0
        while k < self.iterations:
            if candidateStates.isEmpty(): break
            state = candidateStates.pop()
            #print("value iteration for state:", state)
            maxValue = -float("inf")
            for action in self.mdp.getPossibleActions(state):
                QValue = self.computeQValueFromValues(state, action)
               #print("QValue:", QValue)
                if QValue > maxValue: maxValue = QValue
            if self.mdp.isTerminal(state): maxValue = 0
           #print("maxValue:", maxValue)
            self.values[state] = maxValue
            #print("values of states:", self.values)
            predeStates = getPredecessors(state)
            for pre in predeStates:
                diff = calcDiff(pre)
                if diff > self.theta:
                    candidateStates.update(pre, -diff)
            k += 1
            if k >= self.iterations: break
            

