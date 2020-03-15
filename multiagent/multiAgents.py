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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distToFood = 0
        for f in newFood:
            d = util.manhattanDistance(f, newPos)
            distToFood = distToFood + d
        if distToFood != 0:
            distToFood = 1 / distToFood
        else:
            distToFood = 100

        distToGhost = 0
        ghostPos = successorGameState.getGhostPositions()
        for g in ghostPos:
            d2 = util.manhattanDistance(g, newPos)
            distToGhost = distToGhost + d2

        capsulePosition = successorGameState.getCapsules()
        distToCapsule = 0
        for c in capsulePosition:
            d3 = util.manhattanDistance(c, newPos)
            distToCapsule = distToCapsule + d3
        if distToCapsule != 0:
            distToCapsule = 1 / distToCapsule
        else:
            distToCapsule = 100

        numFood = successorGameState.getNumFood()
        if numFood != 0:
            numFood = 1 / numFood

        if newScaredTimes[0] > 0:
            return distToFood + numFood + 5 + successorGameState.getScore() + 100
        elif distToGhost <= 5:
            return distToFood + numFood + distToGhost + distToCapsule + successorGameState.getScore()
        else:
            return distToFood + numFood + 5 + successorGameState.getScore() + 100

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
   
        def helper(state, agentIndex, depth):
            if state.isWin(): 
                return self.evaluationFunction(state)
            if state.isLose():
                return self.evaluationFunction(state)
            if depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                nextAgentIndex = agentIndex + 1
                utilityList = []
                for action in state.getLegalActions(agentIndex):
                    utility = helper(state.generateSuccessor(agentIndex, action), nextAgentIndex, depth)
                    utilityList.append(utility)
                return max(utilityList)

            else:
                nextAgentIndex = agentIndex + 1  
                if state.getNumAgents() == nextAgentIndex:
                    nextAgentIndex = 0
                    depth = depth + 1

                utilityList = []
                for action in state.getLegalActions(agentIndex):
                    utility = helper(state.generateSuccessor(agentIndex, action), nextAgentIndex, depth)
                    utilityList.append(utility)
                return min(utilityList)
        
        maximum = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            utility = helper(gameState.generateSuccessor(0, action), 1, 0)
            if utility > maximum:
                maximum = utility
                bestAction = action

        return bestAction   

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def helper(state, agentIndex, depth, alpha, beta):
            if state.isWin(): 
                return self.evaluationFunction(state)
            if state.isLose():
                return self.evaluationFunction(state)
            if depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                nextAgentIndex = agentIndex + 1
                v = float("-inf")
                for action in state.getLegalActions(agentIndex): 
                    utility = helper(state.generateSuccessor(agentIndex, action), nextAgentIndex, depth, alpha, beta)
                    v = max(v, utility)
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v

            else:
                nextAgentIndex = agentIndex + 1  
                if state.getNumAgents() == nextAgentIndex:
                    nextAgentIndex = 0
                    depth = depth + 1

                v = float("inf")
                for action in state.getLegalActions(agentIndex):
                    utility = helper(state.generateSuccessor(agentIndex, action), nextAgentIndex, depth, alpha, beta)
                    v = min(v, utility)
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v
        
        alpha = float("-inf")
        beta = float("inf")
        maximum = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            utility = helper(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if utility > maximum:
                maximum = utility
                bestAction = action
            if utility > beta:
                return bestAction
            alpha = max(alpha, utility)
        return bestAction 

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
        def helper(state, agentIndex, depth):
            if state.isWin(): 
                return self.evaluationFunction(state)
            if state.isLose():
                return self.evaluationFunction(state)
            if depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                nextAgentIndex = agentIndex + 1
                utilityList = []
                for action in state.getLegalActions(agentIndex):
                    utility = helper(state.generateSuccessor(agentIndex, action), nextAgentIndex, depth)
                    utilityList.append(utility)
                return max(utilityList)

            else:
                nextAgentIndex = agentIndex + 1  
                if state.getNumAgents() == nextAgentIndex:
                    nextAgentIndex = 0
                    depth = depth + 1

                utilityList = []
                v = 0
                numActions = len(state.getLegalActions(agentIndex))
                for action in state.getLegalActions(agentIndex):
                    utility = helper(state.generateSuccessor(agentIndex, action), nextAgentIndex, depth)
                    v = v + (1/numActions) * utility
                return v
        
        maximum = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            utility = helper(gameState.generateSuccessor(0, action), 1, 0)
            if utility > maximum:
                maximum = utility
                bestAction = action

        return bestAction 

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    distToFood = 0
    for f in foodPos:
        d = util.manhattanDistance(f, pos)
        distToFood = distToFood + d
        if distToFood != 0:
            distToFood = 1 / distToFood
        else:
            distToFood = 100

    distToGhost = 0
    ghostPos = currentGameState.getGhostPositions()
    for g in ghostPos:
        d2 = util.manhattanDistance(g, pos)
        distToGhost = distToGhost + d2

    capsulePosition = currentGameState.getCapsules()
    distToCapsule = 0
    for c in capsulePosition:
        d3 = util.manhattanDistance(c, pos)
        distToCapsule = distToCapsule + d3
        if distToCapsule != 0:
            distToCapsule = 1 / distToCapsule
        else:
            distToCapsule = 100

    numFood = currentGameState.getNumFood()
    if numFood != 0:
        numFood = 1 / numFood

    if scaredTimes[0] > 0:
        return distToFood + numFood + 7 + currentGameState.getScore() + 100
    elif distToGhost <= 7:
        return distToFood + numFood + distToGhost + distToCapsule + currentGameState.getScore()
    else:
        return distToFood + numFood + 7 + currentGameState.getScore() + 100

    
    return currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
