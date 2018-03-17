from enum import Enum

# This class stores the enumerate for the labels of the state of a square
# as well as helper functions to print squares.  These enumerated square types will
# also serve as our player labels.
class GameResult(Enum):
    GAME_WIN = 0
    GAME_DRAW = 1
    GAME_LOSE = 2
    GAME_DISQUALIFIED = 3

# This class is essentially in charge of the top level control flow for making a game happen
class MNKGame(object):
    def gameStep(self, board, moving, opponent, history, settings):
        gameOver = False
        winner = None


        # request a move from the agent, then try to make it
        moveX, moveY = moving.move(board, settings)
        history.append(((moveX, moveY), board))
        moveWasLegal = board.addPiece(moveX, moveY, moving.type)
        if settings.verbose:
            print(moving.type, "moves at square: ", (moveX, moveY), board)

        if not moveWasLegal:
            winner = opponent
            gameOver = True
        # Check to see if that move was a winner
        elif board.hasPlayerWon(settings.k, moving.type):
            winner = moving
            gameOver = True

        # first see if we CAN put anymore pieces on the board
        if not board.movesRemain():
            gameOver = True

        return gameOver, winner, moveWasLegal, moveX, moveY

    def playGame(self, board, firstToMove, secondToMove, settings):
        winner = None
        moveWasLegal = True

        firstHistory = []
        secondHistory = []

        # actually play the game
        for _ in range(board._m*board._n):
            gameOver, winner, moveWasLegal, moveX, moveY = self.gameStep(board, firstToMove, secondToMove, firstHistory, settings)
            if gameOver:
                break
            gameOver, winner, moveWasLegal, moveX, moveY = self.gameStep(board, secondToMove, firstToMove, secondHistory, settings)
            if gameOver:
                break

        # hand out rewards to the agents
        if winner == None:
            firstToMove.observeReward(firstHistory, GameResult.GAME_DRAW, settings)
            secondToMove.observeReward(secondHistory, GameResult.GAME_DRAW, settings)

        elif winner == firstToMove and moveWasLegal:
            firstToMove.observeReward(firstHistory, GameResult.GAME_WIN, settings)
            secondToMove.observeReward(secondHistory, GameResult.GAME_LOSE, settings)

        elif winner == firstToMove and not moveWasLegal:
            firstToMove.observeReward(firstHistory, GameResult.GAME_WIN, settings)
            secondToMove.observeReward(secondHistory, GameResult.GAME_DISQUALIFIED, settings)

        elif winner == secondToMove and moveWasLegal:
            firstToMove.observeReward(firstHistory, GameResult.GAME_LOSE, settings)
            secondToMove.observeReward(secondHistory, GameResult.GAME_WIN, settings)

        elif winner == secondToMove and not moveWasLegal:
            firstToMove.observeReward(firstHistory, GameResult.GAME_DISQUALIFIED, settings)
            secondToMove.observeReward(secondHistory, GameResult.GAME_WIN, settings)

        return winner
