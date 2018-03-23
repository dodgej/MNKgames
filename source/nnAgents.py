import torch
import numpy as np
from random import randint
from copy import deepcopy
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
    rewardGameDraw = 1.25
    penaltyLoss = .25
    penaltyIllegalMove = .01
    MAX_TRIES_TO_MOVE = 300

    def __init__(self, squareType, m, n, k):
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

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.conv2_outputs * self.m * self.n)
        x = F.relu(self.fc1(x))

        return self.critic(x), self.actor(x)

    def observeReward(self, history, result, settings):
        Agent.observeReward(self, history, result, settings)

        learningRate = 0.01
        reward = 0
        if result == GameResult.GAME_WIN:
            reward = self.rewardGameWin
        if result == GameResult.GAME_DRAW:
            reward = self.rewardGameDraw
        if result == GameResult.GAME_LOSE:
            reward = self.penaltyLoss
        if result == GameResult.ILLEGAL_MOVE:
            reward = self.penaltyIllegalMove
            learningRate = .1

        optimizer = optim.SGD(self.parameters(), lr=learningRate)


        _, boards = zip(*history)

        for board in boards:

            #FIXME this block is shared with move(). should be abstracted for clarity and maintenance.
            # Compute the NN outputs and move
            nnValue, nnOutput = self(Variable(board.exportToNN()))
            #print("nnoutput", nnOutput)
            actionProbs = F.softmax(nnOutput, dim=1)
            mostProbableAction = actionProbs.max(1)
            # apparently the variable above is a pair, so second element is the argmax
            mostProbableActionIdx = mostProbableAction[1]
            # this is in a variable, so reach in for the tensor and access an item to get an int
            actionChoice = mostProbableActionIdx.data[0]



            #print(nnOutput)

            targetTensor = deepcopy(nnOutput.data)
            newActionProbMass = targetTensor[0][actionChoice] * reward
            rewardPerAction = reward / targetTensor.size()[1]
#            massDiffPerAction = (chosenProbMass - newProbMass)/ (targetTensor.size()[1] - 1)

            targetTensor *= rewardPerAction
            targetTensor[0][actionChoice] = newActionProbMass
            targetVar = Variable(targetTensor)

            loss = nn.SmoothL1Loss()(nnOutput, targetVar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def move(self, board, settings, tries=0):
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
        actionProbs = F.softmax(nnOutput, dim=1)

        mostProbableAction = actionProbs.max(1)
        # apparently the variable above is a pair, so second element is the argmax
        mostProbableActionIdx = mostProbableAction[1]
        # this is in a variable, so reach in for the tensor and access an item to get an int
        actionChoice = mostProbableActionIdx.data[0]

        # determine the board index that action corresponds to
        moveX, moveY = board.convertActionVecToIdxPair(actionChoice)
        #print("move", moveX, " ", moveY)

        if not board.moveIsLegal(moveX, moveY):
            if tries > self.MAX_TRIES_TO_MOVE:
                if settings.verbose:
                    print("****************illegal move selected (", moveX, ",", moveY, ") or [", actionChoice, "]trying again")
                    print(nnOutput)
            else:
                #FIXME option 1 (the one we are taking right now), send a STRONG error signal through the network
                # FIXME option 2: select the next most probable from the actions?
                fakeHistory = [((moveX, moveY), board)]
                self.observeReward(fakeHistory, GameResult.ILLEGAL_MOVE, settings)
                return self.move(board, settings, 1+tries)

        return moveX, moveY
