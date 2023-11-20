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
import math
import statistics


from pacman import GameState
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def value(self, gameState, player,depth):
        if gameState.isWin() or gameState.isLose() or depth == 2:
            return betterEvaluationFunction(gameState)
        if player % 3 == 0:
            if gameState.isWin() or gameState.isLose():
                return betterEvaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)
            v = float('-inf')
            for a in legalActions:
                v = max(self.value(gameState.generateSuccessor(0, a), player + 1,depth), v)
            return v
        else:
            legalActions = gameState.getLegalActions(player % 3)
            v = float('+inf')
            for a in legalActions:
                if player%3 == 2:
                    v = min(self.value(gameState.generateSuccessor(player % 3, a), player + 1,depth+1), v)
                else:
                    v = min(self.value(gameState.generateSuccessor(player % 3, a), player + 1, depth), v)

            return v

    def getAction(self, gameState):
        legalActions = gameState.getLegalActions(0)
        v = float('-inf')
        v_action = legalActions[0]
        player = 0
        for a in legalActions:
            nexts = gameState.generateSuccessor(0, a)
            if self.value(nexts, player+1,0) > v:
                v_action = a
                v = self.value(nexts, player+1,0)

        return v_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def value(self, gameState, player, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 3:
            return betterEvaluationFunction(gameState)
        if player % 3 == 0:
            if gameState.isWin() or gameState.isLose():
                return betterEvaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)
            v = float('-inf')
            for a in legalActions:
                v = max(self.value(gameState.generateSuccessor(0, a), player + 1,depth,alpha,beta), v)
                if v > beta:
                    return v
                alpha = max(alpha,v)
            return v
        else:
            legalActions = gameState.getLegalActions(player % 3)
            v = float('+inf')
            for a in legalActions:
                if player%3 == 2:
                    v = min(self.value(gameState.generateSuccessor(player % 3, a), player + 1,depth+1,alpha,beta), v)
                else:
                    v = min(self.value(gameState.generateSuccessor(player % 3, a), player + 1, depth,alpha,beta), v)
                if v < alpha:
                    return v
                beta = min(beta,v)
            return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        legalActions = gameState.getLegalActions(0)
        v = float('-inf')
        v_action = legalActions[0]
        player = 0
        for a in legalActions:
            nexts = gameState.generateSuccessor(0, a)
            k = self.value(nexts, player+1, 0, float('-inf'), float('+inf'))
            if k > v:
                v_action = a
                v = k

        return v_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def value(self, gameState, player, depth):
        if gameState.isWin() or gameState.isLose() or depth == 3:
            return betterEvaluationFunction(gameState)
        if player % 3 == 0:
            if gameState.isWin() or gameState.isLose():
                return betterEvaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)
            v = float('-inf')
            for a in legalActions:
                v = max(self.value(gameState.generateSuccessor(0, a), player + 1, depth), v)
            return v
        else:
            legalActions = gameState.getLegalActions(player % 3)
            v = 0
            for a in legalActions:
                if player%3 == 2:
                    v += self.value(gameState.generateSuccessor(player % 3, a), player + 1, depth+1)
                else:
                    v += self.value(gameState.generateSuccessor(player % 3, a), player + 1, depth)
            return v/len(legalActions)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        legalActions = gameState.getLegalActions(0)
        v = float('-inf')
        v_action = legalActions[0]
        player = 0
        for a in legalActions:
            nexts = gameState.generateSuccessor(0, a)
            k = self.value(nexts, player+1, 0)
            if k > v:
                v_action = a
                v = k

        return v_action

def betterEvaluationFunction(currentGameState:GameState):
    def minDistanceBfs(currentGameState:GameState):
        walls = currentGameState.getWalls()
        height = 0
        for i in walls:
            height+=1
        width = 0
        for i in walls[0]:
            width += 1

        start_position = currentGameState.getPacmanPosition()
        visited = set()
        queue = util.Queue()
        queue.push([start_position,0])
        while not queue.isEmpty():
            sposition = queue.pop()
            x , y = sposition[0]
            if currentGameState.hasFood(x , y):
                return sposition[1]
            if sposition[0] in visited:
                continue
            visited.add(sposition[0])

            x , y = sposition[0]

            if not walls[x-1][y] and x > 0:
                queue.push([(x-1,y),sposition[1]+1])
            if not walls[x+1][y] and x < height:
                queue.push([(x+1,y),sposition[1]+1])
            if not walls[x][y-1] and y > 0:
                queue.push([(x,y-1),sposition[1]+1])
            if not walls[x][y+1] and y < width:
                queue.push([(x,y+1),sposition[1]+1])
        return float('inf')


    if currentGameState.isWin():
        return 50000 + currentGameState.getScore()
    if currentGameState.isLose():
        return -500000


    numFood = currentGameState.getNumFood()

    return currentGameState.getScore()/5 - 50 * numFood + 9/minDistanceBfs(currentGameState)
# Abbreviation
better = betterEvaluationFunction