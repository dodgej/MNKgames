import torch
import torch.nn as nn
from agent import Agent

#FIXME build a second NN agent that uses an LSTM cell
class cnnAgent(torch.nn.Module, Agent):
    def __init__(self, squareType, k):
        super(cnnAgent, self).__init__()
        #channels = 3 # number of types of square (X, O, empty)
        self.conv1 = nn.Conv2d(self.m, self.n)
        # etc

        numActions = 999 #FIXME the 999 needs to be respecified
        self.critic = nn.Linear(999, 1)
        self.actor = nn.Linear(999, self.m*self.n)

    def forward(self, x):
        x = self.conv1(x)
        # etc

        return self.critic(x), self.actor(x)

    def move(self, board, settings):
        x = 0 # FIXME shape the board into an input
        nnValue, nnMove = self.forward(x)

        # FIXME if it is training, we should capture a bunch of stuff in a dictionary


