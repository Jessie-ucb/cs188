# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

import searchAgents
import search
    
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        """
        print(str(newPos))
        print(str(newFood))
        print(newGhostStates)
        print(str(newScaredTimes))
        """
        "*** YOUR CODE HERE ***"
        " food distance"
        currentFood = currentGameState.getFood()  #This is important!
        foodPos = []
        for x in range(currentFood.width):
            for y in range(currentFood.height):
                if currentFood[x][y] == True:
                    foodPos.append((x,y))
        minDistance = 99999
        for x,y in foodPos:
            foodDistance = abs(newPos[0]-x) + abs(newPos[1]-y)
            if minDistance > foodDistance:
                minDistance = foodDistance
        
        " ghost distance"
        newGhostPos = successorGameState.getGhostPositions()
        ghostDistance = 0
        for x,y in newGhostPos:
            ghostDistance += abs(newPos[0]-x) + abs(newPos[1]-y)
        
        " stop penalty"
        penalty = 0
        if action == Directions.STOP: 
            penalty = 10.0

        " ScaredTimes"
        totalScaredTime = 0
        for time in newScaredTimes: 
            totalScaredTime += time

        " The problem is: when an isolated food is eaten when taking the action, "
        " the newFood will not contain the eaten one ==> could cause the minDistance to be larger than expected (should be 0!) "
        " Solution is, use current Food grid for distance calculating !!!"

        evaluation = successorGameState.getScore() + newFood.count() - 5*minDistance + ghostDistance - penalty + totalScaredTime
        #print("evaluation", evaluation)
        return evaluation
        
        

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def minValue(gameState, agentIndex, actions):
            minV = float("inf")
            legalMoves = gameState.getLegalActions(agentIndex % numAgents)
            minAction = legalMoves[0]
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex % numAgents, action)
                (v, actions) = value(nextState, agentIndex + 1, actions) # agentIndex + 1?
                if v < minV: 
                    minV = v
                    minAction = action
            actions.append(minAction)
            #print("minimizer:", (minV, actions))
            return (minV, actions)

        def maxValue(gameState, agentIndex, actions):
            maxV = -float("inf")
            legalMoves = gameState.getLegalActions(agentIndex % numAgents)
            maxAction = legalMoves[0]
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex % numAgents, action)
                (v, actions) = value(nextState, agentIndex + 1, actions) # agentIndex + 1?
                if v > maxV: 
                    maxV = v
                    maxAction = action
            actions.append(maxAction)
            #print("maximizer:", (maxV, actions))
            return (maxV, actions)

        def value(gameState, agentIndex, actions):
            "terminal state?"
            if agentIndex >= numAgents * self.depth or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState), actions)
            "maximizer"
            if agentIndex % numAgents == 0:
                return maxValue(gameState, agentIndex, actions)
            "minimizers"
            if agentIndex % numAgents > 0:
                return minValue(gameState, agentIndex, actions)
            
        (rootV, rootActions) = value(gameState, 0, [])
        # return actions of depth d(=2), including one maximizer and multiple minimizers
        print("root value:", rootV)
        return rootActions[-1]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def minValue(gameState, agentIndex, actions, minLimit, maxLimit):
            minV = float("inf")
            legalMoves = gameState.getLegalActions(agentIndex % numAgents)
            minAction = legalMoves[0]
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex % numAgents, action)
                (v, actions) = value(nextState, agentIndex + 1, actions, minLimit, maxLimit) # agentIndex + 1?
                if v < minV: 
                    minV = v
                    minAction = action
                if minV < maxLimit:  # alpha-pruning: last agent is maximizer
                    actions.append(action)
                    return(minV, actions)
                if minV < minLimit: minLimit = minV
            actions.append(minAction)
            return (minV, actions)

        def maxValue(gameState, agentIndex, actions, minLimit, maxLimit):
            maxV = -float("inf")
            legalMoves = gameState.getLegalActions(agentIndex % numAgents)
            maxAction = legalMoves[0]
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex % numAgents, action)
                (v, actions) = value(nextState, agentIndex + 1, actions, minLimit, maxLimit) # agentIndex + 1?
                if v > maxV: 
                    maxV = v
                    maxAction = action
                if maxV > minLimit:  # beta-pruning: last agent is minimizer
                    actions.append(action)
                    return(maxV, actions)
                if maxLimit < maxV: maxLimit = maxV
            actions.append(maxAction)
            return (maxV, actions)

        def value(gameState, agentIndex, actions, minLimit, maxLimit):
            "terminal state?"
            if agentIndex >= numAgents * self.depth or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState), actions)
            "maximizer"
            if agentIndex % numAgents == 0:
                return maxValue(gameState, agentIndex, actions, minLimit, maxLimit)
            "minimizers"
            if agentIndex % numAgents > 0:
                return minValue(gameState, agentIndex, actions, minLimit, maxLimit)
            
        (rootV, rootActions) = value(gameState, 0, [], float("inf"), -float("inf"))
        print("root value:", rootV)
        return rootActions[-1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def expectValue(gameState, agentIndex, actions):
            expectV = 0
            legalMoves = gameState.getLegalActions(agentIndex % numAgents)
            #minAction = legalMoves[0]
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex % numAgents, action)
                (v, actions) = value(nextState, agentIndex + 1, actions) # agentIndex + 1?
                expectV += v
            expectV /= len(legalMoves)
            #actions.append(minAction)
            #print("minimizer:", (minV, actions))
            return (expectV, actions)

        def maxValue(gameState, agentIndex, actions):
            maxV = -float("inf")
            legalMoves = gameState.getLegalActions(agentIndex % numAgents)
            maxAction = legalMoves[0]
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex % numAgents, action)
                (v, actions) = value(nextState, agentIndex + 1, actions) # agentIndex + 1?
                if v > maxV: 
                    maxV = v
                    maxAction = action
            actions.append(maxAction)
            #print("maximizer:", (maxV, actions))
            return (maxV, actions)

        def value(gameState, agentIndex, actions):
            "terminal state?"
            if agentIndex >= numAgents * self.depth or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState), actions)
            "maximizer"
            if agentIndex % numAgents == 0:
                return maxValue(gameState, agentIndex, actions)
            "expects"
            if agentIndex % numAgents > 0:
                return expectValue(gameState, agentIndex, actions)
            
        (rootV, rootActions) = value(gameState, 0, [])
        # return actions of depth d(=2), including one maximizer and multiple minimizers
        print("root value:", rootV)
        return rootActions[-1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    " number of foods"
    numFoods = currentGameState.getFood().count()
    " closet food "
    searchAgent = searchAgents.ClosestDotSearchAgent('breadthFirstSearch')
    actions = searchAgent.findPathToClosestDot(currentGameState) # Find a path
    distanceToFood = 0
    if actions: distanceToFood = len(actions)
    " all ghosts"
    pacmanPos = currentGameState.getPacmanPosition()
    ghostPos = currentGameState.getGhostPositions()    
    minGhostDistance = 0
    for pos in ghostPos:
        pos = (int(pos[0]), int(pos[1]))
        ghostDistance = searchAgents.mazeDistance(pacmanPos, pos, currentGameState)
        if ghostDistance < minGhostDistance: minGhostDistance = ghostDistances
    " scared time"
    ghostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    totalScaredTime = 0
    for time in newScaredTimes: 
        totalScaredTime += time
    #if totalScaredTime > 0: minGhostDistance = -minGhostDistance
    return currentGameState.getScore() -numFoods*2 + 1/(distanceToFood+1) -2/(minGhostDistance+1) +totalScaredTime

# Abbreviation
better = betterEvaluationFunction
