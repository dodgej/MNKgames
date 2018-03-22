import torch
import numpy as np
from random import randint
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from agent import Agent
from mnkgame import GameResult

# a useful resource
# https://github.com/greydanus/visualize_atari

# TODO cnn with lstm
#One purely CNN
class cnnAgent(Agent, torch.nn.Module):
    # network reward function parameters
    rewardGameWin = 2
    rewardGameDraw = 1
    penaltyLoss = -1
    penaltyIllegalMove = -10000

    def __init__(self, squareType, m, n, k):
        print("Initializing network...")
        Agent.__init__(self, squareType, m, n, k)
        torch.nn.Module.__init__(self)

        # network parameters
        input_channels = 3
        self.kernelSize = 3
        self.conv1_outputs = 11
        self.conv2_outputs = 13
        boardSize = self.m * self.n
        self.conv1 = nn.Conv2d(input_channels, self.conv1_outputs, self.kernelSize, padding=int(self.kernelSize/2))
        self.conv2 = nn.Conv2d(self.conv1_outputs, self.conv2_outputs, self.kernelSize, padding=int(self.kernelSize/2))
        self.fc1 = nn.Linear(self.conv2_outputs * self.m * self.n, boardSize)

        self.critic = nn.Linear(boardSize, 1)
        self.actor = nn.Linear(boardSize, boardSize)

        print("Initialed network")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.conv2_outputs * self.m * self.n)
        x = F.relu(self.fc1(x))

        return self.critic(x), self.actor(x)

    def observeReward(self, history, result, settings):
        print("Observing reward...")
        Agent.observeReward(self, history, result, settings)

        reward = 0
        if result == GameResult.GAME_WIN:
            reward = self.rewardGameWin
        if result == GameResult.GAME_DRAW:
            reward = self.rewardGameDraw
        if result == GameResult.GAME_LOSE:
            reward = self.penaltyLoss
        if result == GameResult.ILLEGAL_MOVE:
            reward = self.penaltyIllegalMove

        print("making optimizer... ")
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.8)

        _, boards = zip(*history)

        for board in boards:
            # FIXME may want to do this operation elsewhere
            optimizer.zero_grad()

            nnValue, nnOutput = self(Variable(board.exportToNN()))
            #print("nnoutput", nnOutput)
            #FIXME the making of the loss object is currently the main thing I am not sure on. This should not be seg faulting...
            print("generating loss  (occasionally seg faults here)")
            loss = nn.CrossEntropyLoss(reduce=False)(nnOutput, Variable(torch.LongTensor(-reward)))
            print("loss generated")

            loss.backward()
            print("BACKWARD pass complete")
            optimizer.step()
            print("Optimizer step taken")

        print("reward Observed")

    def move(self, board, settings):
        print("moving...")
        '''
        for k, v in self.state_dict().items():
            print("Layer {}".format(k))
            print(v)
        '''
        #first verify that a move exists
        if not board.movesRemain():
            print("requested move on full board!")
            return None, None

        nnValue, nnOutput = self.forward(Variable(board.exportToNN()))
        actionProbs = F.softmax(nnOutput)

        mostProbableAction = actionProbs.max(1)
        # apparently the variable above is a pair, so second element is the argmax
        mostProbableActionIdx = mostProbableAction[1]
        # this is in a variable, so reach in for the tensor and access an item to get an int
        actionChoice = mostProbableActionIdx.data[0]

        # determine the board index that action corresponds to
        moveX, moveY = board.convertActionVecToIdxPair(actionChoice)
        #print("move", moveX, " ", moveY)

        if not board.moveIsLegal(moveX, moveY):
            print("illegal move selected, trying again")

            #FIXME option 1, send a STRONG error signal through the network
            fakeHistory = [((moveX, moveY), board)]
            self.observeReward(fakeHistory, GameResult.ILLEGAL_MOVE, settings)
            #FIXME careful here, dont want infinite recursion
            return self.move(board, settings)


            #FIXME option 2: select the next most probable from the actions?

        print("moved")
        return moveX, moveY
