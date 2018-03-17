from board import Board
from square import Square
from mnkgame import MNKGame
from agent import RandomAgent, RandomIllegalAgent
from searchAgents import SearchAgent
from nnAgents import cnnAgent

# This class stores settings for ALL the constituent parts for the pipeline
class Settings(object):
    def __init__(self):
        # Board parameters
        self.m = 6
        self.n = 7
        self.k = 5

        # Agent parameters.  Specify different agent types here via constructor
        self.Xagent = RandomAgent(Square.X_HAS, self.k)
        self.Oagent = SearchAgent(Square.O_HAS, self.k)
        self.numGamesToEstimateValue = 5

        # Outermost loop parameters
        self.numGamesToTest = 1
        self.verbose = True


# this function plays a number of games stored in the settings and reports results
def playGames(settings):
    Xwins = 0.0
    Xloses = 0.0
    draws = 0.0

    print("****Testing ", settings.numGamesToTest, " games of agents looking for sequences of length k=", settings.k, " using ", settings.numGamesToEstimateValue, " games to estimate value")

    for i in range(settings.numGamesToTest):
        # new game, create a fresh board
        if settings.verbose:
            print("Creating a M x N board, where m =", settings.m, " and n=", settings.n, "\n")
        board = Board(settings.m, settings.n)

        # play the game, taking turns being first to act
        if i % 2 == 0:
            winner = MNKGame().playGame(board, settings.Xagent, settings.Oagent, settings)
        else:
            winner = MNKGame().playGame(board, settings.Oagent, settings.Xagent, settings)

        # do the bookkeeping now that a result is obtained
        if winner == settings.Xagent:
            Xwins += 1
            if settings.verbose:
                print("X emerges victorious over the vile O!!!!! in game ", i)
        elif winner == settings.Oagent:
            Xloses += 1
            if settings.verbose:
                print("O has defeated the disgusting X!!!!! in game ", i)
        elif winner == None:
            draws += 1
            if settings.verbose:
                print("fought to a draw... maybe next time. In game ", i)

    # All games complete, generate some final output
    if settings.verbose:
        print("Xwins=", Xwins, "Xloses=", Xloses, "draws=", draws, )

if __name__ == '__main__':
    settings = Settings()
    playGames(settings)
