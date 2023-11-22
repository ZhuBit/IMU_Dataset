import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets as dset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset

from dataset import IMUDatasetResample
from models import MS_TCN, MS_TCN2
from models import focal_loss, tmse_loss

import torchmetrics

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

Epochs = 1000 #500  # 100

Lr_Rate = 0.001  # 0.001
batch_size = 32 #256 #128  # 32 #4 8 16

print_progress = 20

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

data_dir = 'C:/Projects/01_A2P/HAR/06_Table_Dataset/annotated_IMUs/'
train_dataset = IMUDatasetResample(data_dir, freq=25, sample_len=10)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dir = 'C:/Projects/01_A2P/HAR/06_Table_Dataset/annotated_IMUs/' # 'C:/Projects/01_A2P/HAR/06_Table_Dataset/annotated_IMUs/'
val_dataset = IMUDatasetResample(data_dir, freq=25, sample_len=10)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# net = MS_TCN(num_stages=10, num_layers=4, num_f_maps=128, dim=12, num_classes=10)
net = MS_TCN2(num_layers_PG=11, num_layers_R=10, num_R=4, num_f_maps=128, dim=12, num_classes=10)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=Lr_Rate, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Epochs//10, gamma=0.9)

train_loss = list()
train_acc = list()

val_loss = list()
val_acc = list()

train_prog = 0
val_prog = 0

for ep in range(1, Epochs):
    for iteration, data in enumerate(train_dataloader):

        net.train()

        left_data = data[0].to(device)
        right_data = data[1].to(device)
        hands_data = torch.cat([left_data, right_data], dim=1)

        labels = data[2].type(torch.long).to(device)

        #hands = data[3]

        outputs = net(hands_data)[-1]

        loss = 0
        splits = labels.shape[0]

        for i, value in enumerate(outputs):
            # loss += ce_loss(value.permute(1,0), labels[i])
            loss += focal_loss(value.permute(1, 0), labels[i], num_class=10, alpha=-1, gamma=1)
            loss += tmse_loss(value, labels[i], gamma=0.1)

        loss = 10 * loss / splits

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())
        train_prog += 1

        acc = 0
        for i, value in enumerate(outputs):
            prob = F.softmax(value.permute(1, 0))
            pred = prob.data.max(dim=1)[1]
            # print()
            # print(pred)
            # print(labels[i])
            # print(torchmetrics.functional.accuracy(pred, labels[i]).item())
            acc += torchmetrics.functional.accuracy(pred, labels[i], task="multiclass", num_classes=10).item()
        train_acc.append(acc / splits)

        with torch.no_grad():
            val_data = next(iter(val_dataloader))
            net.eval()

            left_data = data[0].to(device)
            right_data = data[1].to(device)
            hands_data = torch.cat([left_data, right_data], dim=1)

            labels = data[2].type(torch.long).to(device)

            outputs = net(hands_data)[-1]

            loss_val = 0
            splits = labels.shape[0]

            for i, value in enumerate(outputs):
                # loss_val += ce_loss(value.permute(1, 0), labels[i])
                loss_val += focal_loss(value.permute(1, 0), labels[i], num_class=10, alpha=-1, gamma=1)
                loss_val += tmse_loss(value, labels[i], gamma=0.1)

            loss_val = 10 * loss_val / splits
            val_prog += 1

            acc_val = 0
            for i, value in enumerate(outputs):
                prob = F.softmax(value.permute(1, 0))
                pred = prob.data.max(dim=1)[1]
                acc_val += torchmetrics.functional.accuracy(pred, labels[i], task="multiclass", num_classes=10).item()

            for j in range(iteration + 1):
                # len(dataset_split['train']) // len(dataset_split['test'])):  # same length of the lists...
                val_loss.append(loss_val.item())
                val_acc.append(acc_val / splits)

        if ep % 10 == 0:
            curr_t_loss = sum(train_loss[-train_prog:]) / train_prog
            curr_t_acc = sum(train_acc[-train_prog:]) / train_prog
            # curr_v_loss = 0
            # curr_v_acc = 0
            curr_v_loss = sum(val_loss[-val_prog:]) / val_prog
            curr_v_acc = sum(val_acc[-val_prog:]) / val_prog
            print(
                '[%d/%d][%d/%d]\tTraining Loss: %.4f \tTraining Accuracy: %.4f \tValidation Loss: %.4f \tValidation Accuracy: %.4f'
                % (
                ep, Epochs, iteration, int(len(train_dataloader) / batch_size) + 1, curr_t_loss, curr_t_acc, curr_v_loss,
                curr_v_acc))
            train_prog = 0
            val_prog = 0

        if ep % 100 == 0:
            net_path = './MS-TCN_Vol1_e' + str(ep) + '.pt'
            torch.save(net.state_dict(), net_path)

