from enum import Enum

# This class stores the enumerate for the labels of the state of a square
# as well as helper functions to print squares.  These enumerated square types will
# also serve as our player labels.
class Square(Enum):
    OPEN = 0
    X_HAS = 1
    O_HAS = 2

    # this function handles print
    def __repr__(self):
        if self.value == Square.OPEN.value:
            return "-"
        elif self.value == Square.X_HAS.value:
            return "X"
        elif self.value == Square.O_HAS.value:
            return "O"
        else:
            print(self.value)
            print("PANIC!!!! UNKNOWN TYPE in repr")
            return ""

    # this function handles calls to str()
    def __str__(self):
        if self.value == Square.OPEN.value:
            return "-"
        elif self.value == Square.X_HAS.value:
            return "X"
        elif self.value == Square.O_HAS.value:
            return "O"
        else:
            print(self.value)
            print("PANIC!!!! UNKNOWN TYPE in repr")
            return ""