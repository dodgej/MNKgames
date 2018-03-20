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
    penaltyIllegalMove = -1 #FIXME might want this a lot higher if we can be trickier in backprop

    def __init__(self, squareType, m, n, k):
        Agent.__init__(self, squareType, m, n, k)
        torch.nn.Module.__init__(self)
        # need these in constructor, will be appended onto during experience to do the backprop
        self.histories = []
        self.rewards = []

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
        #print('Input tensor size: ', x.size())
        x = F.relu(self.conv1(x))
        #print('after C1 tensor size: ', x.size())
        x = F.relu(self.conv2(x))
        #print('after C2 tensor size: ', x.size())
        x = x.view(-1, self.conv2_outputs * self.m * self.n)
        #print('after view: ', x.size())
        x = F.relu(self.fc1(x))
        #print('after linear: ', x.size())

        return self.critic(x), self.actor(x)

    def observeReward(self, history, result, settings):
        Agent.observeReward(self, history, result, settings)
        self.histories.append(history)

        if result == GameResult.GAME_WIN:
            self.rewards.append(self.rewardGameWin)
        if result == GameResult.GAME_DRAW:
            self.rewards.append(self.rewardGameDraw)
        if result == GameResult.GAME_LOSE:
            self.rewards.append(self.penaltyLoss)
        if result == GameResult.GAME_DISQUALIFIED:
            self.rewards.append(self.penaltyIllegalMove)

    def train(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        #print("HISTORIES", self.histories)
        listOfExportedBoardHistories = []

        # FIXME may want to do this operation elsewhere
        optimizer.zero_grad()
        rewardIndex = -1

        for hist in self.histories:
            rewardIndex += 1
            moves, boards = zip(*hist)

            # FIXME may want to slice off the last move of the action history if it was illegal
            #  (dont want to punish the legal and potentially GOOD moves made prior to cheating)

            for board in boards:
                nnValue, nnOutput = self(Variable(board.exportToNN()))
                #print("nnoutput", nnOutput)
                loss = nn.CrossEntropyLoss()(nnOutput, Variable(torch.LongTensor(-self.rewards[rewardIndex])))

            loss.backward()
            optimizer.step()

        self.histories = []
        self.rewards = []

    def move(self, board, settings):
        nnValue, nnOutput = self.forward(Variable(board.exportToNN()))
        #print("Move output", nnOutput)
        probs = F.softmax(nnOutput)

        #print("probabilities", probs)
        action = probs.max(1)[1].data
        #print("action", action)
        moveX, moveY = board.convertActionVecToIdxPair(action[0])
        #print("move", moveX, " ", moveY)

        return moveX, moveY

#cnn with filter size = k
#FIXME weird error. self.k is not callable
class cnnAgent2(Agent, torch.nn.Module):
    # network reward function parameters
    rewardGameWin = 2
    rewardGameDraw = 1
    penaltyLoss = -1
    penaltyIllegalMove = -10000

    def __init__(self, squareType, m, n, k):
        Agent.__init__(self, squareType, m, n, k)
        torch.nn.Module.__init__(self)
        # need these in constructor, will be appended onto during experience to do the backprop
        self.histories = []
        self.rewards = []

        # network parameters
        input_channels = 3
        kernelSize = 3
        conv1_outputs = 6
        boardSize = self.m * self.n
        self.conv1 = nn.Conv2d(input_channels, conv1_outputs, self.k)
        self.conv2 = nn.Conv2d(conv1_outputs, 16, self.k)
        self.fc1 = nn.Linear(16 * (10-2*((self.k)-1)) * (10-2*((self.k)-1)), boardSize)

        self.critic = nn.Linear(boardSize, 1)
        self.actor = nn.Linear(boardSize, boardSize)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        print('Input tensor size: ')
        print(x.size())
        pad = nn.ZeroPad2d((10-self.n, 0, 10-self.m, 0))
        x = pad(x)
        x = F.relu(self.conv1(x))
        print(x.size())
        x = F.relu(self.conv2(x))
        print(x.size())
        x = x.view(-1, 16 * (10-2*(self.k-1)) * (10-2*(self.k-1)))
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())

        return self.critic(x), self.actor(x)

    def observeReward(self, history, result, settings):
        Agent.observeReward(self, history, result, settings)
        self.histories.append(history)

        if result == GameResult.GAME_WIN:
            self.rewards.append(self.rewardGameWin)
        if result == GameResult.GAME_DRAW:
            self.rewards.append(self.rewardGameDraw)
        if result == GameResult.GAME_LOSE:
            self.rewards.append(self.penaltyLoss)
        if result == GameResult.GAME_DISQUALIFIED:
            self.rewards.append(self.penaltyIllegalMove)

    def train(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        moves, boards = zip(*self.histories)

        #FIXME may want to do this operation elsewhere
        optimizer.zero_grad()

        outputs = self(boards)
        loss = nn.CrossEntropyLoss()(outputs, -self.rewards)

        loss.backward()
        optimizer.step()

        self.histories = []
        self.rewards = []



    def move(self, board, settings):
        nnValue, nnMove = self.forward(Variable(board.exportToNN()))
        print(nnMove)
        probs = F.softmax(nnMove, dim=0)

        print(probs)
        action = probs.max(1)[1].data
        print(action)
        #board.convertActionVecToIdxPair()

        #FIXME rather than computing an index pair, just make a fixed move until things work better.
        return 0,0

#cnn with m and n. no padding
class cnnAgent3(Agent, torch.nn.Module):
    # network reward function parameters
    rewardGameWin = 2
    rewardGameDraw = 1
    penaltyLoss = -1
    penaltyIllegalMove = -10000

    def __init__(self, squareType, m, n, k):
        Agent.__init__(self, squareType, m, n, k)
        torch.nn.Module.__init__(self)
        # need these in constructor, will be appended onto during experience to do the backprop
        self.histories = []
        self.rewards = []

        # network parameters
        input_channels = 3
        kernelSize = 3
        conv1_outputs = 6
        boardSize = self.m * self.n
        t_m = self.m -4
        t_n = self.n -4
        self.conv1 = nn.Conv2d(input_channels, conv1_outputs, kernelSize)
        self.conv2 = nn.Conv2d(conv1_outputs, 16, kernelSize)
        self.fc1 = nn.Linear(16 * t_m * t_n, boardSize)

        self.critic = nn.Linear(boardSize, 1)
        self.actor = nn.Linear(boardSize, boardSize)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        print('Input tensor size: ')
        print(x.size())
        #pad = nn.ZeroPad2d((10-self.n, 0, 10-self.m, 0))
        #x = pad(x)
        x = F.relu(self.conv1(x))
        print(x.size())
        x = F.relu(self.conv2(x))
        print(x.size())
        x = x.view(-1, 16 * (10-2*(self.k-1)) * (10-2*(self.k-1)))
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())

        return self.critic(x), self.actor(x)

    def observeReward(self, history, result, settings):
        Agent.observeReward(self, history, result, settings)
        self.histories.append(history)

        if result == GameResult.GAME_WIN:
            self.rewards.append(self.rewardGameWin)
        if result == GameResult.GAME_DRAW:
            self.rewards.append(self.rewardGameDraw)
        if result == GameResult.GAME_LOSE:
            self.rewards.append(self.penaltyLoss)
        if result == GameResult.GAME_DISQUALIFIED:
            self.rewards.append(self.penaltyIllegalMove)

    def train(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        moves, boards = zip(*self.histories)

        #FIXME may want to do this operation elsewhere
        optimizer.zero_grad()

        outputs = self(boards)
        loss = nn.CrossEntropyLoss()(outputs, -self.rewards)

        loss.backward()
        optimizer.step()

        self.histories = []
        self.rewards = []



    def move(self, board, settings):
        nnValue, nnMove = self.forward(Variable(board.exportToNN()))
        print(nnMove)
        probs = F.softmax(nnMove, dim=0)

        print(probs)
        action = probs.max(1)[1].data
        print(action)
        #board.convertActionVecToIdxPair()

        #FIXME rather than computing an index pair, just make a fixed move until things work better.
        return 0,0
