import torch
import torch.nn as nn
import torch.nn.functional as F

class VLAPolicy(nn.Module):
    def __init__(self, visual_dim, quality_dim, num_actions):
        super().__init__()
        input_dim = visual_dim + quality_dim
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_actions)
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(256)
        self.ln4 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(p=0.1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vfeat, qfeat):
        # concat along feature dim
        x = torch.cat([vfeat, qfeat], dim=-1)

        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = self.ln4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = F.relu(x)

        return self.out(x)