from random import randint
import abc
from copy import deepcopy
from square import Square
import mnkgame
import numpy as np

# This is the abstract base class (hence the use of the abc import) that specifies an agent
class Agent(object):
    __metaclass__ = abc.ABCMeta

    # agents ALL need to know whether they are playing X or O,
    # as well as the sequence length they seek to create
    def __init__(self, squareType, k):
        self.type = squareType
        self.k = k

    # This specifies that all Agents must specify a move function to override this one
    @abc.abstractmethod
    def move(self, board, settings): pass


# This agent simply behaves randomly (within the rules).
# Search based agents will use the random agents to estimate value
class RandomAgent(Agent):
    def move(self, board, settings):
        openSquares = board.getOpenSquares()
        if [] == openSquares:
            print("PANIC, selecting from 0 options")
        return openSquares[randint(0, len(openSquares) - 1)]


# This agent considers each possible move by estimating the value by playing a fixed number of random games from
# the board position resulting from playing that move on the current board
class SearchAgent(Agent):
    # FIXME extend to pure MCTS agent. Beam search also worth considering
    def move(self, board, settings):
        # These will be a bunch of synchronous arrays connecting a move (square) to win/loss results
        openSquares = board.getOpenSquares()
        estimatedWins = []
        estimatedLosses = []
        estimatedDraws = []

        # Create the random agents that will play to estimate board state values
        myAgent = RandomAgent(self.type, self.k)
        oppType = Square.O_HAS
        if oppType == self.type:
            oppType = Square.X_HAS
        oppAgent = RandomAgent(oppType, self.k)

        # Copy the settings and make sure the new one is not set to verbose (too much output)
        newSettings = deepcopy(settings)
        newSettings.verbose = False
        # for each possible move, play a bunch of games to estimate its value
        for x,y in openSquares:
            wins = 0.0
            losses = 0.0
            draws = 0.0
            for i in range(settings.numGamesToEstimateValue):
                # copy the "current" board and make the move we are considering
                newBoard = deepcopy(board)
                newBoard.addPiece(x, y, self.type)

                # play a random game from here
                winner = mnkgame.MNKGame().playGame(newBoard, oppAgent, myAgent, newSettings)

                # do bookkeeping based on who won
                if winner == myAgent:
                    wins += 1.0
                elif winner == oppAgent:
                    losses += 1.0
                else:
                    draws += 1.0

            # Stick results on the synchronous arrays to unpack later.
            estimatedWins.append(wins)
            estimatedLosses.append(losses)
            estimatedDraws.append(draws)

        if settings.verbose:
            print("*** Simulation complete, results: (square, wins, losses, draws)")
            for square, wins, losses, draws in zip(openSquares, estimatedWins, estimatedLosses, estimatedDraws):
                 print(square, wins, losses, draws)

        # return the move that had highest win rate
        return openSquares[np.argmax(estimatedWins)] #FIXME possibly break ties by prob to lose?


# this will be the one(many?) we want to do well (by combining search and deep learning)
class HybridAgent(Agent):
    # FIXME this should get the board represented as some # of channels for the deep model, then request a move
    # from a deep network. Varying the representation is a parameter we should investigate
    def move(self, board, settings):
        openSquares = board.getOpenSquares()
        return openSquares[randint(0, len(openSquares) - 1)]