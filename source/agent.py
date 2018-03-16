from random import randint
import abc

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



# this will be the one(many?) we want to do well (by combining search and deep learning)
class HybridAgent(Agent):
    # FIXME this should get the board represented as some # of channels for the deep model, then request a move
    # from a deep network. Varying the representation is a parameter we should investigate
    def move(self, board, settings):
        openSquares = board.getOpenSquares()
        return openSquares[randint(0, len(openSquares) - 1)]