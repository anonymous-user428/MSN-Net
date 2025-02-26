import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import DataLoader
import math

class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, num_subclusters=16):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.n_subclusters = num_subclusters
        self.s = math.sqrt(2) * math.log(num_classes*num_subclusters - 1)
        self.W = Parameter(torch.FloatTensor(num_classes*num_subclusters, num_features))
        # self.W.requires_grad = False
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, mixup: torch.tensor, label: torch.tensor):
        assert mixup.size(0) == label.size(0) and input.size(0) == label.size(0), "Error - label and input size not match!"
        repeated_label = torch.repeat_interleave(label, self.n_subclusters, dim=1)
        # normalize features
        x = F.normalize(input, dim=1)
        # normalize weights
        W = F.normalize(self.W, dim=1)
        # print(W.cpu().detach().numpy().tolist())
        # print(W.shape)
        # dot product
        logits = F.linear(x, W)
        # print(f"max W: {torch.max(W.cpu().detach()).item()} max logits: {torch.max(logits.cpu().detach())}")
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        B, C = repeated_label.size()
        # one_hot = torch.zeros_like(label)
        # for i in range(B):
        #     if mixup[i] == 0:
        #         one_hot[i][repeated_label[i].argmax()] = 1
        #     else: # if mixup, keep the two classes
        #         one_hot[i][repeated_label[i].topk(2).indices] = 1
        with torch.no_grad():
            max_logit = torch.max(self.s*logits)
            # B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.exp(self.s*logits-max_logit)
            B_avg = torch.sum(B_avg) / logits.size(0)
            # print()
            # _theta = torch.where(one_hot<1, torch.zeros_like(logits), theta)*repeated_label
            mixup_index = torch.nonzero(mixup, as_tuple=True)[0]
            selected_theta = torch.index_select(theta*repeated_label, 0, mixup_index).sum(dim=1)
            # non_mixup_index = torch.nonzero(1-mixup, as_tuple=True)[0]
            # selected_theta[non_mixup_index] = torch.index_select(_theta, 0, non_mixup_index).sum()
            theta_med = torch.quantile(selected_theta, q=0.5, interpolation='midpoint')
            self.s = (torch.log(B_avg)+max_logit) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        
        # print(" max_logit: {:.3f}, B_avg: {:.3f}, theta_med: {:.3f}, s: {:.3f}".format(max_logit.item(), B_avg.item(), theta_med.item(), self.s.item()))
        logits *= self.s
        out = F.softmax(logits, dim=-1)
        out = (out.view(-1, self.n_classes, self.n_subclusters)).sum(dim=-1)
        return out

