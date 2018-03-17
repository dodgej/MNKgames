import torch
from random import randint
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from agent import Agent
from mnkgame import GameResult

# a useful resource
# https://github.com/greydanus/visualize_atari

#FIXME build 3 agents.  One purely CNN, one CNN with filter size = k, one with an LSTM cell
class cnnAgent(Agent, torch.nn.Module):
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
        self.conv1 = nn.Conv2d(input_channels, conv1_outputs, kernelSize)
        self.conv2 = nn.Conv2d(conv1_outputs, 16, kernelSize)
        self.fc1 = nn.Linear(16 * kernelSize * kernelSize, boardSize)

        self.critic = nn.Linear(boardSize, 1)
        self.actor = nn.Linear(boardSize, boardSize)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 3 * 3) #FIXME deal with these magic numbers at some point
        x = F.relu(self.fc1(x))

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

        #FIXME shape problems from before are making this function broken. Hammer out architecture first
        print(probs)
        action = probs.max(1)[1].data
        print(action)
        #board.convertActionVecToIdxPair()

        #FIXME rather than computing an index pair, just make a fixed move until things work better.
        return 0,0




