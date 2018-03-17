from square import Square
import torch

# This class stores the board and which pieces are on it
class Board(object):
    def __init__(self, m, n):
        self._m = m
        self._n = n
        # this is now to be indexed [m][n]. Python doesnt really do 2d arrays, but nested lists are close enough.
        self._theBoard = [[Square(Square.OPEN) for j in range(n)] for i in range(m)]

    def convertActionVecToIdxPair(self, actionLongVec):
        yIdx = int(actionLongVec / self._n)
        xIdx = actionLongVec % self._n
        return xIdx, yIdx

    # This function attempts to add a piece to the board. If the square is occupied, it complains.
    # returns whether the move was legal
    def addPiece(self, mIndex, nIndex, type):
        if self._theBoard[mIndex][nIndex] == Square.OPEN:
            self._theBoard[mIndex][nIndex] = type
            return True
        else:
            print("*********ERROR! attempt to add to an occupied square")
            return False

    # returns whether or not any moves remain
    def movesRemain(self): #FIXME cache results and update via bookkeeping, dont do N^2 work per call
        for i in range(self._m):
            for j in range(self._n):
                if self._theBoard[i][j] == Square.OPEN:
                    return True
        return False

    # returns a list containing index pairs for each empty square on the board
    def getOpenSquares(self):
        result = []
        for i in range(self._m):
            for j in range(self._n):
                if self._theBoard[i][j] == Square.OPEN:
                    result.append((i,j))
        return result

    # returns whether or not a player has a sequence of the specified length on the board
    def hasPlayerWon(self, sequenceLength, player):
        # Scan m dimension
        for i in range(self._m-sequenceLength+1):
            for j in range(self._n):
                for k in range(sequenceLength):
                    if(self._theBoard[i+k][j] != player):
                        break
                    elif k == sequenceLength - 1:
                        return True

        # Scan n dimension
        for i in range(self._m):
            for j in range(self._n-sequenceLength+1):
                for k in range(sequenceLength):
                    if(self._theBoard[i][j+k] != player):
                        break
                    elif k == sequenceLength - 1:
                        return True

        # Scan diagonal from upper left to lower right
        for i in range(self._m-sequenceLength+1):
            for j in range(self._n-sequenceLength+1):
                for k in range(sequenceLength):
                    if(self._theBoard[i+k][j+k] != player):
                        break
                    elif k == sequenceLength - 1:
                        return True

        # Scan diagonal from lower left to upper right
        for i in range(sequenceLength-1, self._m):
            for j in range(self._n-sequenceLength+1):
                for k in range(sequenceLength):
                    if(self._theBoard[i-k][j+k] != player):
                        break
                    elif k == sequenceLength - 1:
                        return True

        # if we got here, no one has won yet
        return False

    def exportToNN(self):
        channels = 3
        batchSize = 1
        boardTensor = torch.zeros(batchSize, channels, self._m, self._n)

        # doctor up the tensor with current board state
        for i in range(self._m):
            for j in range(self._n):
                if self._theBoard[i][j] == Square.OPEN:
                    boardTensor[0][0][i][j] = 1
                elif self._theBoard[i][j] == Square.O_HAS:
                    boardTensor[0][1][i][j] = 1
                elif self._theBoard[i][j] == Square.X_HAS:
                    boardTensor[0][2][i][j] = 1
                else:
                    print("Panic: dont know what this square type means")

        return boardTensor

    # This function handles printing boards
    def __repr__(self):
        result = "\n"
        for i in range(self._m):
            for j in range(self._n):
                result += str(self._theBoard[i][j])
            result += "\n"
        return result