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
        
        #Initializes the current and new position's information
        currentFood = currentGameState.getFood()
        currentGhostPotisions = currentGameState.getGhostPositions()
        currentPacmanPosition = currentGameState.getPacmanPosition()
        currentFoodList = currentFood.asList()
        currentFoodDistances = []
        currentGhostDistances = []

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostPotisions = successorGameState.getGhostPositions()
        newFood = successorGameState.getFood()
        newPacmanPosition = successorGameState.getPacmanPosition()
        newFoodList = newFood.asList()
        newGhostPotisions = successorGameState.getGhostPositions()
        newFoodDistances = []
        newGhostDistances = []

        score = 0

        #If there is no more food return score
        if len(newFoodList) == 0:
            return score

        #If the pacman will meet a ghost return negative score
        if newPacmanPosition in [ghostPositions for ghostPositions in newGhostPotisions ]:
            score -= 1
            return 
        
        #If the pacman will eat a food return positive score
        if newPacmanPosition in [foodPositions for foodPositions in currentFoodList]:
            score +=1
            return score

        #Calculates the distance between the current position of pacman with every food
        for food in currentFoodList:
            currentFoodDistances.append(manhattanDistance(currentPacmanPosition,food))

        #Calculates the distance between the current position of pacman with every ghost
        for ghost in currentGhostPotisions:
            currentGhostDistances.append(manhattanDistance(currentPacmanPosition,ghost))

        #Calculates the distance between the new position of pacman with every food
        for food in newFoodList:
            newFoodDistances.append(manhattanDistance(newPacmanPosition,food))

        #Calculates the distance between the new position of pacman with every ghost
        for ghost in newGhostPotisions:
            newGhostDistances.append(manhattanDistance(newPacmanPosition,ghost))

        #Mininum distance to current Food
        minCurrFoodDist= min(currentFoodDistances)

        #Minimum distance to current Ghost
        minCurrGhostDist = min(currentGhostDistances)

        #Minimum distance to new Food
        minNewFoodDist= min(newFoodDistances)

        #Minimum distance to new Ghost
        minNewGhostDist = min(newGhostDistances)


        #If pacman comes closer to food add 1 else subtract 1
        if minNewFoodDist > minCurrFoodDist:
            score -= 1
        elif minNewFoodDist < minCurrFoodDist:
            score += 1

        #If pacman goes away from ghost add 1 else substract 1
        if minNewGhostDist > minCurrGhostDist:
            score += 1
        elif minNewGhostDist > minCurrGhostDist:
            score -=1

        return score

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

        #Gets the legal pacman's actions
        legalActions = gameState.getLegalActions(0)                            
        newGameStates = []
        scores = []

        #Generate every legal successor state
        for action in legalActions:
            newGameStates.append(gameState.generateSuccessor(0, action))

        #Start tha minimax algorithm for every state    
        for newState in newGameStates:
            scores.append(self.minValue(newState, 0, 1)) 

        bestScore = max(scores)

        #Return the action for the best score
        return legalActions[scores.index(bestScore)]

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def maxValue(self,gameState,currentDepth):
        #Evaluates the game state if minimax reached on max depth or the state is win/lose
        if (self.depth == currentDepth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)

        value = -9999999.0

        #Gets the legal pacman's actions
        legalActions = gameState.getLegalActions(0)
        newGameStates = []

        #Generate every legal successor state
        for action in legalActions:
            newGameStates.append(gameState.generateSuccessor(0, action))

        #Return the max value of this state's minimax
        value = max([self.minValue(newState, currentDepth, 1) for newState in newGameStates])
        return value

    def minValue(self,gameState,currentDepth,ghost):
        #Evaluates the game state if minimax reached on max depth or the state is win/lose
        if (self.depth == currentDepth  or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)

        value = 9999999.0
        numAgents = gameState.getNumAgents()
        ghostValues = []
        newGameStates = []

        #Gets the legal pacman's actions
        legalActions = gameState.getLegalActions(ghost)
        for action in legalActions:
           newGameStates.append(gameState.generateSuccessor(ghost, action))

        """
        If there is more ghost agents for every next-state of the current ghost
        call the minValue for the remaining ghost
        Else increase the depth and call the maxValue to continue the minimax algorithm
        """
        if (ghost < numAgents - 1):
            for newState in newGameStates:
                ghostValues.append(self.minValue(newState, currentDepth, ghost+1))
        else:
            for newState in newGameStates:
                ghostValues.append(self.maxValue(newState, currentDepth+1))

        #Return the min value
        value = min(ghostValues)
        return value



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = -999999
        beta = 999999
        scores = []
        bestScore = -9999999

        #Gets the legal pacman's actions
        legalActions = gameState.getLegalActions(0)

        """
        For each legal action generates the successor state,
        adds the result of minValue function and higher value of scores to bestcore
        if the bestscore is higher than beta will prun and return the bestscore's action
        else alpha will take the higher value of (alpha,bestScore)
        """
        for action in legalActions:
            newState = gameState.generateSuccessor(0, action)
            scores.append(self.minValue(newState,0,1,alpha,beta)) 
            bestScore = max(scores[-1],bestScore)

            if bestScore > beta:
                return legalActions[scores.index(bestScore)]
            alpha = max(alpha,bestScore)

        #At the end returns the action of bestScore
        return legalActions[scores.index(bestScore)]

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def maxValue(self,gameState,currentDepth,alpha,beta):
        #Evaluates the game state if minimax reached on max depth or the state is win/lose
        if (self.depth == currentDepth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)

        max_value = -99999
        #Gets the legal pacman's actions        
        legalActions = gameState.getLegalActions(0)

        """
        For each legal action generates the successor state,
        sets the max_value to the max(max_value,MINVALUE)
        if the max_value is higher than beta will prun and return the max_value
        else alpha will take the higher value of (alpha,max_value)
        """        
        for action in legalActions:
            newState = gameState.generateSuccessor(0, action)

            max_value = max(max_value,self.minValue(newState,currentDepth,1,alpha,beta))

            if max_value > beta:
                return max_value
            alpha = max(alpha,max_value)
        #At the end returns the max_value
        return max_value

    def minValue(self,gameState,currentDepth,ghost,alpha,beta):
        #Evaluates the game state if minimax reached on max depth or the state is win/lose
        if (self.depth == currentDepth  or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)

        min_value = 99999
        
        numAgents = gameState.getNumAgents()

        #Get the legal pacman's actions
        legalActions = gameState.getLegalActions(ghost)

        """
        For every ghost's action
            generate the successor state
            if there are remaining ghosts, call the minValue for the next ghost
                If the returned min_value is smaller than alpha return the min_value(prun the next actions)
                else set beta the min of beta,min_value
            else call the maxValue of the next depth and set the min of (min_value,maxValue) to min_value
                if the min_value is smaller than alpha return the min_value(prun the next actions)
                else sets beta to min of (beta,min_value)
        return the min_value
        """
        for action in legalActions:
            newState = gameState.generateSuccessor(ghost, action)

            if (ghost < numAgents - 1):
                min_value = min(min_value,self.minValue(newState,currentDepth,ghost+1,alpha,beta))

                if min_value < alpha:
                    return min_value
                beta = min(beta,min_value)                

            else:
                min_value = min(min_value,self.maxValue(newState,currentDepth+1,alpha,beta))

                if min_value < alpha:
                    return min_value
                beta = min(beta,min_value)

        return min_value
        util.raiseNotDefined()

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
        newGameStates = []
        scores = []

        #Get the legal pacman's actions
        legalActions = gameState.getLegalActions(0)
        
        #Generates every legal successor state
        for action in legalActions:
            newGameStates.append(gameState.generateSuccessor(0, action))
        
        #For each new state save the expecti value to scores
        for newState in newGameStates:
            scores.append(self.expectiValue(newState,0,1))

        #Return the action with the highest score
        bestScore = max(scores)
        return legalActions[scores.index(bestScore)]

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def maxValue(self,gameState,currentDepth):
        #Evaluates the game state if minimax reached on max depth or the state is win/lose
        if (self.depth == currentDepth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)

        max_value = -99999999.0
        #Get the legal pacman's actions
        legalActions = gameState.getLegalActions(0)
        newGameStates = []

        #For each legal action generates the successor state
        for action in legalActions:
            newGameStates.append(gameState.generateSuccessor(0, action))
        
        #For each successor state
        #sets the max_value to the max(max_value,expectiValue())
        for newState in newGameStates:
            max_value = max(max_value,self.expectiValue(newState,currentDepth,1))
        
        return max_value

    def expectiValue(self,gameState,currentDepth,ghost):
        #Evaluates the game state if minimax reached on max depth or the state is win/lose
        if (self.depth == currentDepth  or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)

        e_value = 0.0
        numAgents = gameState.getNumAgents()
        ghostValues = []
        newGameStates = []

        #Get the legal pacman's actions
        legalActions = gameState.getLegalActions(ghost)
        
        #Generates the new game States
        for action in legalActions:
            newGameStates.append(gameState.generateSuccessor(ghost, action))

        """
        If there are remaining ghosts call the min value for every ghost's state
        and add the returning value to the ghostValues
        else call the maxValue for the next depth
        """
        if (ghost < numAgents - 1):
            for newState in newGameStates:
                ghostValues.append(self.expectiValue(newState,currentDepth,ghost+1))

        else:
            for newState in newGameStates:
                ghostValues.append(self.maxValue(newState,currentDepth+1))

        #Calculate the average ghostValues and return it
        for g_value in ghostValues:
            e_value += g_value
        e_value = e_value/len(ghostValues)
        return e_value

def manhattanDistance( xy1, xy2 ):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )

def betterEvaluationFunction(currentGameState):
    #Get some infos about the GameState
    pacmanPosition = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()
    food = currentGameState.getFood()
    foodList = food.asList()                
    capsuleList = currentGameState.getCapsules()
    score = currentGameState.getScore()

    foodDistances = []
    ghostDistances = []

    """
    If there is no more food left return the current score + 500(winning points)
    Subtract the number of Foods*10 from score
    Subtract the number of capsules*9 from score
    For each ghost
        If it is scared subtract its distance from score (pacman will hunt it)
        else check if the pacman is close enough to die on the next move so return -10000
            to make pacman run away
    Find the distances to each food and subtract the minum from the score
    return the score

    """

    if len(foodList) == 0:
        return score+500


    score -= len(foodList) * 10
    score -= len(capsuleList) * 9

    for ghost in ghosts:

        if ghost.scaredTimer:
            score -= manhattanDistance(pacmanPosition, ghost.getPosition())
        elif manhattanDistance(pacmanPosition, ghost.getPosition()) < 1:
            return -10000

    for i_food in foodList:
        foodDistances.append(manhattanDistance(pacmanPosition, i_food))
  
    score -= min(foodDistances)

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

