from enum import Enum

# This class stores the enumerate for the labels of the state of a square
# as well as helper functions to print squares.  These enumerated square types will
# also serve as our player labels.
class GameResult(Enum):
    GAME_WIN = 0
    GAME_DRAW = 1
    GAME_LOSE = 2
    ILLEGAL_MOVE = 3

# This class is essentially in charge of the top level control flow for making a game happen
class MNKGame(object):
    def gameStep(self, board, moving, opponent, history, settings):
        gameOver = False
        winner = None

        # request a move from the agent, then try to make it
        moveX, moveY = moving.move(board, settings)
        moveWasLegal = board.addPiece(moveX, moveY, moving.type)
        if settings.verbose:
            print(moving.type, "makes a Legal=", moveWasLegal, " move @ square: ", (moveX, moveY), board)

        # Check to see if that move was a winner
        if board.hasPlayerWon(settings.k, moving.type):
            winner = moving
            gameOver = True

        # first see if we CAN put anymore pieces on the board
        if not board.movesRemain():
            gameOver = True

        history.append(((moveX, moveY), board))
        return gameOver, winner, moveX, moveY

    def playGame(self, board, firstToMove, secondToMove, settings):
        winner = None
        moveWasLegal = True

        firstHistory = []
        secondHistory = []

        # actually play the game
        for _ in range(board._m*board._n):
            gameOver, winner, moveX, moveY = self.gameStep(board, firstToMove, secondToMove, firstHistory, settings)
            if gameOver:
                break
            gameOver, winner, moveX, moveY = self.gameStep(board, secondToMove, firstToMove, secondHistory, settings)
            if gameOver:
                break

        #FIXME but possibly only do this if the agent is in training mode
        # hand out rewards to the agents
        if winner == None:
            firstToMove.observeReward(firstHistory, GameResult.GAME_DRAW, settings)
            secondToMove.observeReward(secondHistory, GameResult.GAME_DRAW, settings)

        elif winner == firstToMove:
            firstToMove.observeReward(firstHistory, GameResult.GAME_WIN, settings)
            secondToMove.observeReward(secondHistory, GameResult.GAME_LOSE, settings)

        elif winner == secondToMove:
            firstToMove.observeReward(firstHistory, GameResult.GAME_LOSE, settings)
            secondToMove.observeReward(secondHistory, GameResult.GAME_WIN, settings)

        return winner
