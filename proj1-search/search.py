# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def backtrackActions(start, end, parents): # only works for undirected graph!!
    state = end
    actions = util.Stack()
    while not state == start:
        for p in parents:
            if p[0] == state:
                lastState = p[1]
                actions.push(p[2])
        state = lastState

    actionsList = []
    while not actions.isEmpty():
        actionsList.append(actions.pop())
    #print("Backtracked actions are:", actionsList)
    return actionsList

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    "*** YOUR CODE HERE ***"
    fringes = util.Queue()
    fringes.push(problem.getStartState()) # list of states/vars
    visited = {problem.getStartState()}     # set of states/vars         
    parents = set()  # map storing the parent for each node, set of paired tuples
    
    def dfsHelper(problem, state, visited, parents):
        actions = []
        if problem.isGoalState(state):
            actions = backtrackActions(problem.getStartState(), state, parents)
            return actions
        successors = problem.getSuccessors(state)     # reverse the order of successors ? Why ?
        for nextState, action, cost in successors[::-1]:
            if nextState not in visited:
                parents.add((nextState,state,action))
                visited.add(nextState)
                actions = dfsHelper(problem, nextState, visited, parents)  
                if not actions == []:
                    break                   
        return actions
        
    return dfsHelper(problem, problem.getStartState(), visited, parents)
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"  
    fringes = util.Queue()
    fringes.push(problem.getStartState()) # list of states/vars
    visited = set()
    visited.add(problem.getStartState())  # set of states/vars
    parents = set()  # map storing the parent for each node, set of paired tuples
    while not fringes.isEmpty():
        state = fringes.pop()
        if problem.isGoalState(state):
            #print("Reach goal!")
            return backtrackActions(problem.getStartState(), state, parents)
        successors = problem.getSuccessors(state)     # reverse the order of successors ? Why ?
        for nextState, action, cost in successors[::-1]:
            if nextState not in visited:
                fringes.push(nextState)
                visited.add(nextState)
                parents.add((nextState,state,action))
    
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringes = util.PriorityQueue()
    fringes.update((problem.getStartState(),problem.getStartState(),None), 0)
    visited = set()
    parents = set()
    totalCost = 0
    while not fringes.isEmpty():
        (state, lastState, action) = fringes.pop() # add lastState's totalCost, don't backtracking everytime, cause it's very time-consuming
        if state not in visited:
            visited.add(state)   # location for visited checking is different from DFS/BFS? 
            if not action == None: # exception: start, NOT GOOD to use IF !!
                parents.add((state,lastState,action))
                actions = backtrackActions(problem.getStartState(), state, parents)
                totalCost = problem.getCostOfActions(actions)
            if problem.isGoalState(state):
                return backtrackActions(problem.getStartState(), state, parents)
            successors = problem.getSuccessors(state)     # reverse the order of successors ? Why ?
            for nextState, action, cost in successors[::-1]:
                if nextState not in visited:
                    fringes.update((nextState,state,action), totalCost+cost)
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringes = util.PriorityQueue()
    fringes.update((problem.getStartState(),problem.getStartState(),None), 0)
    visited = set()
    parents = set()
    totalCost = 0
    while not fringes.isEmpty():
        (state, lastState, action) = fringes.pop()
        if state not in visited:
            visited.add(state)
            if not action == None: # exception: start, NOT GOOD to use IF
                parents.add((state,lastState,action))
                actions = backtrackActions(problem.getStartState(), state, parents)
                totalCost = problem.getCostOfActions(actions)
            if problem.isGoalState(state):
                return backtrackActions(problem.getStartState(), state, parents)
            successors = problem.getSuccessors(state)     # reverse the order of successors ? Why ?
            for nextState, action, cost in successors[::-1]:
                if nextState not in visited:
                    fringes.update((nextState,state,action), totalCost+cost+heuristic(nextState, problem))
    

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
