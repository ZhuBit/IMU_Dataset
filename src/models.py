# https://github.com/yabufarha/ms-tcn/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.bn = nn.BatchNorm1d(12) #nn.LazyBatchNorm1d()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in
             range(num_stages - 1)])

    def forward(self, x):
        x = self.bn(x)
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(out)  # * mask[:, 0:1, :], mask) #s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs #[-1]


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


from torchvision.ops import sigmoid_focal_loss
def focal_loss(x, y, num_class, alpha=0.25, gamma=2):
    x_squeeze = x.squeeze(0).transpose(0, 1)
    targ = torch.nn.functional.one_hot(y, num_classes=num_class).type(torch.float)
    #print('X shape: ', x_squeeze.shape, 'Y shape: ', targ.shape)
    #print('x_squeeze:', x_squeeze)
    return sigmoid_focal_loss(x_squeeze, targ, alpha=alpha, gamma=gamma, reduction='mean')

"""def focal_loss(x, y, num_class, alpha=0.25, gamma=2):
    # Assuming y has shape [batch_size, sequence_length]
    # and x has shape [num_classes, batch_size, sequence_length]
    print('Y shape: ', y)
    targ = torch.nn.functional.one_hot(y, num_classes=num_class).type(torch.float)  # [batch_size, sequence_length, num_classes]
    targ = targ.permute(2, 0, 1)  # [num_classes, batch_size, sequence_length]

    return sigmoid_focal_loss(x, targ, alpha=alpha, gamma=gamma, reduction='mean')"""

# https://github.com/yiskw713/asrf/blob/main/libs/loss_fn/tmse.py
def tmse_loss(x, targ, gamma=0.15):
    total_loss = 0.0

    loss = F.mse_loss(F.log_softmax(x[:, 1:], dim=1), F.log_softmax(x[:, :-1], dim=1), reduction='none')

    loss = torch.clamp(loss, min=0, max=4 ** 2)
    total_loss += torch.mean(loss) #torch.sum(loss)

    return total_loss * gamma

class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2*num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out

