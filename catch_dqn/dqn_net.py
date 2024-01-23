import torch.nn as nn


class DQNet(nn.Module):
    """ DQN network """
    def __init__(self, num_actions, fps):
        super(DQNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(fps, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        """ Forward pass """
        output_conv_layers = self.conv_layers(x).view(x.size()[0], -1)
        return self.fc_layers(output_conv_layers)
