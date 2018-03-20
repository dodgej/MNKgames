from agent import Agent, RandomAgent
from square import Square
from copy import deepcopy
from square import Square
import mnkgame
import numpy as np

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
        myAgent = RandomAgent(self.type, self.m, self.n, self.k)
        oppType = Square.O_HAS
        if oppType == self.type:
            oppType = Square.X_HAS
        oppAgent = RandomAgent(oppType, self.m, self.n, self.k)

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

        # compute a scoring function for each move based on wins/losses/draws
        moveScores = [W * 2 - L + D for W,L,D in zip(estimatedWins, estimatedLosses, estimatedDraws)]

        '''
        if settings.verbose:
            print("*** Simulation complete, results: (square, wins, losses, draws, SCORE)")
            for square, wins, losses, draws, score in zip(openSquares, estimatedWins, estimatedLosses, estimatedDraws, moveScores):
                 print(square, wins, losses, draws, score)
        '''

        # select the move with the best score
        return openSquares[np.argmax(moveScores)]
