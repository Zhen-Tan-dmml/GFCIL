import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution


class GPN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GPN_Valuator(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Valuator, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.fc3(x)

        return x


class Classifier(nn.Module):
    def __init__(self, nhid, nclass):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, x):
        x = self.fc(x)
        return x


class Attention(nn.Module):
    def __init__(self, base_class_num, nway, dropout):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(base_class_num, base_class_num // 2)
        self.fc2 = nn.Linear(base_class_num // 2, nway)
        self.fc3 = nn.Linear(nway, 1)
        self.dropout = dropout

    def forward(self, base_prototye, seen_prototye, novel_prototye):
        x0 = F.relu(self.fc1(base_prototye))
        x0 = F.dropout(x0, self.dropout, training=self.training)
        x0 = F.relu(self.fc2(x0))

        if seen_prototye is not None:
            x1 = torch.cat((x0, seen_prototye, novel_prototye), 0)
        else:
            x1 = torch.cat((x0, novel_prototye), 0)
        x1 = F.relu(self.fc3(x1))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = torch.squeeze(x1, -1)

        x2 = F.relu(self.fc3(novel_prototye))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = torch.squeeze(x2, -1)

        x2 = torch.transpose(x2, 0, 1)
        atts = F.softmax(torch.mm(x1, x2), dim=0)
        atts = torch.squeeze(atts)

        return atts





