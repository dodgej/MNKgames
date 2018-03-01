# This class is essentially in charge of the top level control flow for making a game happen
class MNKGame(object):

    def playGame(self, board, firstToMove, secondToMove, settings):
        # check to see if this board is already won
        if board.hasPlayerWon(settings.k, firstToMove.type):
            return firstToMove
        if board.hasPlayerWon(settings.k, secondToMove.type):
            return secondToMove

        # Make moves until someone wins
        while (True):
            # test to see if any moves exist
            if not board.movesRemain():
                return None

            # Make move for the first to act
            x1, y1 = firstToMove.move(board, settings)
            board.addPiece(x1, y1, firstToMove.type)
            if settings.verbose:
                print(firstToMove.type, "moves at square: ", (x1, y1), board)
            # Check to see if that move was a winner
            if board.hasPlayerWon(settings.k, firstToMove.type):
                return firstToMove

            # whether or not the last move won, the board could now be drawn, test for that.
            if not board.movesRemain():
                return None

            # moves remain, second to act makes a move!
            x2, y2 = secondToMove.move(board, settings)
            board.addPiece(x2, y2, secondToMove.type)
            if settings.verbose:
                print(secondToMove.type, "moves at square: ", (x2, y2), board)
            # Check to see if that move was a winner
            if board.hasPlayerWon(settings.k, secondToMove.type):
                return secondToMove
