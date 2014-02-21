# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions, Actions
import random, util
from game import Agent
import sys
import util


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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        "*** YOUR CODE HERE ***"


        foodList = newFood.asList();
        newGhostPosition = successorGameState.getGhostPositions()
        #the distance from pacman to each ghost
        foodDist = [manhattanDistance(food, newPos) for food in foodList]
        ghostDistance = [manhattanDistance(newPos, ghost) for ghost in newGhostPosition]
        if currentGameState.getPacmanPosition() == newPos:
            return -1000000

        for gd in ghostDistance:
           if gd < 2:
                return -1000000

        if len(foodDist) == 0:
            return 1000000
        else:
            minfoodDist = min(foodDist)
            maxfoodDist = max(foodDist)

        return 1000/sum(foodDist) + 10000/len(foodDist)


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
        """
        "*** YOUR CODE HERE ***"
        def maxplayer(gameState, depth):
            if depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            actionList = gameState.getLegalActions(0)
            bestScore = -sys.maxint - 1
            bestAction = None

            if len(actionList) == 0:
                return (self.evaluationFunction(gameState), None)

            for action in actionList:
                newState = gameState.generateSuccessor(0, action)
                newScore = minplayer(newState, 1, depth)[0]
                if (newScore > bestScore):
                    bestScore, bestAction = newScore, action
            return (bestScore, bestAction)



        def minplayer(gameState, ID, depth):
            actionList = gameState.getLegalActions(ID)
            bestScore = sys.maxint
            bestAction = None

            if len(actionList) == 0:
                return (self.evaluationFunction(gameState), None)

            for action in actionList:
                newState = gameState.generateSuccessor(ID, action)
                if (ID == gameState.getNumAgents() - 1):
                    newScore = maxplayer(newState, depth + 1)[0]
                else:
                    newScore = minplayer(newState, ID + 1, depth)[0]

                if (newScore < bestScore):
                    bestScore, bestAction = newScore, action
            return (bestScore, bestAction)

        return maxplayer(gameState, 0)[1]





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxplayer(gameState, depth, alpha, beta):
            if depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            actionList = gameState.getLegalActions(0)
            bestScore = -sys.maxint - 1
            bestAction = None

            if len(actionList) == 0:
                return (self.evaluationFunction(gameState), None)

            for action in actionList:
                if (alpha > beta):
                    return (bestScore, bestAction)
                newState = gameState.generateSuccessor(0, action)
                newScore = minplayer(newState, 1, depth, alpha, beta)[0]
                if (newScore > bestScore):
                    bestScore, bestAction = newScore, action
                if (newScore > alpha):
                    alpha = newScore
            return (bestScore, bestAction)



        def minplayer(gameState, ID, depth, alpha, beta):
            actionList = gameState.getLegalActions(ID)
            bestScore = sys.maxint
            bestAction = None

            if len(actionList) == 0:
                return (self.evaluationFunction(gameState), None)

            for action in actionList:
                if (alpha > beta):
                    return (bestScore, bestAction)

                newState = gameState.generateSuccessor(ID, action)
                if (ID == gameState.getNumAgents() - 1):
                    newScore = maxplayer(newState, depth + 1, alpha, beta)[0]
                else:
                    newScore = minplayer(newState, ID + 1, depth, alpha, beta)[0]

                if (newScore < bestScore):
                    bestScore, bestAction = newScore, action
                if (newScore < beta):
                    beta = newScore
            return (bestScore, bestAction)

        return maxplayer(gameState, 0, -sys.maxint - 1, sys.maxint)[1]



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
        def maxplayer(gameState, depth):
            if depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            actionList = gameState.getLegalActions(0)
            bestScore = - sys.maxint - 1
            bestAction = None

            if len(actionList) == 0:
                return (self.evaluationFunction(gameState), None)

            for action in actionList:
                newState = gameState.generateSuccessor(0, action)
                newScore = randplayer(newState, 1, depth)[0]
                if (newScore > bestScore):
                    bestScore, bestAction = newScore, action
            return (bestScore, bestAction)



        def randplayer(gameState, ID, depth):
            actionList = gameState.getLegalActions(ID)
            totalScore = 0
            bestAction = None

            if len(actionList) == 0:
                return (self.evaluationFunction(gameState), None)

            for action in actionList:
                newState = gameState.generateSuccessor(ID, action)
                if (ID == gameState.getNumAgents() - 1):
                    newScore = maxplayer(newState, depth + 1)[0]
                else:
                    newScore = randplayer(newState, ID + 1, depth)[0]
                totalScore += newScore/len(actionList)
            return (totalScore, bestAction)

        return maxplayer(gameState, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: in the evaluation function we are trying to combine the
      safety of the pacman and the state of the food to assign a heuristic
      value to the currentGameState. Below are some of the features used:

          1. Empty spaces around each food left. This is aimed at keeping the
             food in a block, and allow the pacman to clear them efficiently

          2. manhattanDistance to the closest ghost, this is to determine how
             safe the pacman is. Also allows pacman to eat Ghosts when they are
             scared.(Different safety level is reflected by the safety parameter

          3. Inverse of the closest manhattanDistance to any food. Want to maximaze
             this value to minimize distance

      Use a combination of this three main feature to assign value to the gameState


    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Pos = (int(Pos[0]), int(Pos[1]))
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()

    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    wallList = currentGameState.getWalls().asList()
    foodList = Food.asList()
    GhostPosition = currentGameState.getGhostPositions()

    #the distance from pacman to each ghost
    foodDist = [manhattanDistance(Pos, food) for food in foodList]
    foodDist = sorted(foodDist)

    ########## keep food in the same Block  ###########
    totalEmp = 0
    for food in foodList:
        #foodCounter[tuple(food)] = manhattanDistance(Pos, food)
        totalEmp += EmptyAround(food, Food, wallList)

    ###################################################

    ############Safety of the Pacman############
    ghostDistance = [manhattanDistance(Pos, (int(ghost[0]), int(ghost[1]))) for ghost in GhostPosition]
    safety = 0

    if sum(ScaredTimes) > 3:
        for gd in ghostDistance:
            if gd < 2:
                safety += 0.5
            elif gd < 3:
                safety += 0.2
            elif gd < 4:
                safety += 0.08
            elif gd < 6:
                safety += 0.04
            elif gd < 11:
                safety -= 0.03
    else:
        for gd in ghostDistance:
            if gd < 2:
                safety -= 0.5
            elif gd < 3:
                safety -= 0.2
            elif gd < 4:
                safety -= 0.08
            elif gd < 6:
                safety -= 0.04
            elif gd < 11:
                safety += 0.03

    for time in ScaredTimes:
        safety += time

    ##########################################




    #########Follow the food #################
    foodie = currentGameState.getScore()

    inversefoodDist = 0

    if len(foodDist) > 0:
        inversefoodDist = 1.0/min(foodDist)

    foodie += (min(ghostDistance) * (inversefoodDist**2) - totalEmp * 6.5)

    ##########################################


    return foodie + safety




def square(x):
    return x * x

def EmptyAround(pos, foodState, wallList):

    x,y = pos
    num_emp = 0
    ls = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    counter = 0
    while counter < 4:
        i , j = ls[counter]
        if foodState[x + i][y + j] == False and (x + i, y + j) not in wallList:
            num_emp += 1
        counter += 1
    return num_emp

better = betterEvaluationFunction
