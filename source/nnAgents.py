import torch
from random import randint
import torch.nn as nn
from agent import Agent
from mnkgame import GameResult

# a useful resource
# https://github.com/greydanus/visualize_atari

#FIXME build 3 agents.  One purely CNN, one CNN with filter size = k, one with an LSTM cell
class cnnAgent(torch.nn.Module, Agent):
    rewardGameWin = 2
    rewardGameDraw = 1
    penaltyLoss = -1
    penaltyIllegalMove = -10000

    def __init__(self, squareType, k):
        super(cnnAgent, self).__init__()
        # #channels = 3 # number of types of square (X, O, empty)
        # self.conv1 = nn.Conv2d(self.m, self.n)
        # # etc
        #
        # numActions = 999 #FIXME the 999 needs to be respecified
        # self.critic = nn.Linear(999, 1)
        # self.actor = nn.Linear(999, self.m*self.n)

        # need these in constructor, will be appended onto during experience to do the backprop
        self.histories = []
        self.rewards = []

    def observeReward(self, history, result):
        self.histories.append(history)

        if result == GameResult.GAME_WIN:
            self.rewards.append(self.rewardGameWin)
        if result == GameResult.GAME_DRAW:
            self.rewards.append(self.rewardGameDraw)
        if result == GameResult.GAME_LOSE:
            self.rewards.append(self.penaltyLoss)
        if result == GameResult.GAME_DISQUALIFIED:
            self.rewards.append(self.penaltyIllegalMove)
        #FIXME call the superclass's fn?
        #FIXME when we backprop gradients from these results, we need to clear the containers


    def forward(self, x):
        #x = self.conv1(x)
        # FIXME etc

        #return self.critic(x), self.actor(x)
        return None

    def move(self, board, settings):
        #x = 0 # FIXME shape the board into an input
        #nnValue, nnMove = self.forward(x)

        openSquares = board.getOpenSquares()
        if [] == openSquares:
            print("PANIC, selecting from 0 options")
        return openSquares[randint(0, len(openSquares) - 1)]


