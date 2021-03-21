# install package for displaying animation
# !pip install JSAnimation

# custom utilies for displaying animation, collecting rollouts and more
import pong_utils
import torch
import gym
import time

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# check which device is being used.
# I recommend disabling gpu until you've made sure that the code runs
device = torch.device("cpu")
# device = pong_utils.device
print("using device: ", device)


# render ai gym environment
# PongDeterministic does not contain random frameskip
# so is faster to train than the vanilla Pong-v4 environment
env = gym.make('PongDeterministic-v4')

print("List of available actions: ", env.unwrapped.get_action_meanings())

# we will only use the actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5
# the 'FIRE' part ensures that the game starts again after losing a life
# the actions are hard-coded in pong_utils.py

# PREPROCESSING
# show what a preprocessed image looks like
env.reset()
_, _, _, _ = env.step(0)
# get a frame after 20 steps
for _ in range(20):
    frame, _, _, _ = env.step(1)

plt.subplot(1, 2, 1)
plt.imshow(frame)
plt.title('original image')

plt.subplot(1, 2, 2)
plt.title('preprocessed image')

# 80 x 80 black and white image
plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
plt.show()

# ########
# POLICY #
# ########

# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        ########
        #  Modify your neural network
        ########

        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride
        # (round up if not an integer)

        # output = 20x20 here
        self.conv = nn.Conv2d(2, 1, kernel_size=4, stride=4)
        self.size = 1*20*20

        # 1 fully connected layer
        self.fc = nn.Linear(self.size, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        #  #######
        #  Modify your neural network
        #  #######
        x = F.relu(self.conv(x))
        # flatten the tensor
        x = x.view(-1, self.size)
        return self.sig(self.fc(x))


# use your own policy!
# policy=Policy().to(device)
policy = pong_utils.Policy().to(device)

# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
