


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import backend
import gridworld


import random, util, math
import numpy as np
import copy


class QLearningAgent(ReinforcementAgent):

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

          
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        
        return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
          
        return self.qValues[(state, action)] 

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, it returns a value of 0.0.
        """
          
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        return max(self.getQValue(state, a) for a in legalActions) 

    def computeActionFromQValues(self, state): 
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        it returns None.
        """
          
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        maxQ = self.computeValueFromQValues(state)

        bestActions = [a for a in legalActions if self.getQValue(state, a) == maxQ]

        return random.choice(bestActions)

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, we 
        choose None as the action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
          
        if not legalActions:
            return action

        if util.flipCoin(self.epsilon): 
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward: float):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        """
          
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args["epsilon"] = epsilon
        args["gamma"] = gamma
        args["alpha"] = alpha
        args["numTraining"] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman. 
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent
    """

    def __init__(self, extractor="IdentityExtractor", **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
         return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
          
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0.0
        for feature, value in features.items():
            q_value += self.weights[feature] * value
        return q_value

    def update(self, state, action, nextState, reward: float):
        """
         update the weights based on transition
        """
          
        features = self.featExtractor.getFeatures(state, action)
        # dif (target - prediction)
        next_value = self.computeValueFromQValues(nextState)  # max_a' Q(s', a')
        difference = (reward + self.discount * next_value) - self.getQValue(state, action)

        # on met a jour les poids
        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value
 
    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        if self.episodesSoFar == self.numTraining:

              
            print("Weights learned by ApproximateQAgent:")
            for feature, weight in self.weights.items():
                print(f"{feature}: {weight:.4f}")
            pass
