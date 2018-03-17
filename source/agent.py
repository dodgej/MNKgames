from random import randint
import abc

from mnkgame import GameResult

# This is the abstract base class (hence the use of the abc import) that specifies an agent
class Agent(object):
    __metaclass__ = abc.ABCMeta

    # agents ALL need to know whether they are playing X or O,
    # as well as the sequence length they seek to create
    def __init__(self, squareType, m, n, k):
        self.type = squareType
        self.m = m
        self.n = n
        self.k = k


    def observeReward(self, history, result, settings):
        if settings.verbose:
            if result == GameResult.GAME_WIN:
                print("Agent ", self.type, " has observed a WIN reward")
            if result == GameResult.GAME_DRAW:
                print("Agent ", self.type, " has observed a DRAW reward")
            if result == GameResult.GAME_LOSE:
                print("Agent ", self.type, " has observed a LOSS penalty")
            if result == GameResult.GAME_DISQUALIFIED:
                print("Agent ", self.type, " has observed a DQ penalty")
            for i in range(0, len(history)-1):
                print(history[i])

    def train(self):
        pass

    # This specifies that all Agents must specify a move function to override this one
    @abc.abstractmethod
    def move(self, board, settings): pass


# This agent simply behaves randomly (within the rules).
class RandomAgent(Agent):
    def move(self, board, settings):
        openSquares = board.getOpenSquares()
        if [] == openSquares:
            print("PANIC, selecting from 0 options")
        return openSquares[randint(0, len(openSquares) - 1)]

# This agent simply behaves randomly, irrespective of the rules.
class RandomIllegalAgent(Agent):
    def move(self, board, settings):
        randomSquareLongVec = randint(0, board._m*board._n - 1)
        print("Randomly selected square ", randomSquareLongVec)
        return board.convertActionVecToIdxPair(randomSquareLongVec)