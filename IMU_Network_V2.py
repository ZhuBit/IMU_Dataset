import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import SlidingWindowIMUsDataset
from src.models import MS_TCN2, MS_TCN
from src.models import focal_loss, tmse_loss

#testing transformer
from src.transformer_models import ASformer

import torchmetrics
import warnings
import csv
import pandas as pd
import math
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=UserWarning)

def train(device, train_dataloader, val_dataloader, net, optimizer, scheduler, Epochs, print_progress, writer):
    train_loss = list()
    train_acc = list()
    val_loss = list()
    val_acc = list()

    for ep in range(1, Epochs + 1):
        train_prog = 0
        val_prog = 0

        for iteration, (features, labels) in enumerate(train_dataloader):
            net.train()
            print(f'------------------------Training Epoch: {ep} \tIteration: {iteration}-----------------------------------------')
            features = features.to(device).permute(0,2,1)
            #print(f'Input Features: {features.shape} \tLabels: {labels.shape}')

            labels = labels.to(device)

            optimizer.zero_grad()

            #testing transformer network
            outputs = net(features, torch.ones(features.shape[0], 2, features.shape[2]).to(device))[-1]

            #outputs = net(features)[-1]
            if torch.isnan(features).any():
                raise ValueError("NaN detected in 'features'")
            if torch.isnan(labels).any():
                raise ValueError("NaN detected in 'labels'")
            if torch.isnan(outputs).any():
                raise ValueError("NaN detected in 'outputs'")
            loss = 0
            splits = labels.shape[0]
            print('**************************outputs.shape[0], labels[0].shape: ', outputs.shape[0], labels[0].shape)
            for i in range(outputs.shape[0]):  # Looping through the first dimension (5 in this case)
                output_reshaped = outputs[i].squeeze()  # Reshaping to [37, 10]
                #print(f'Train Output Shape: {output_reshaped.shape} \tLabels Shape: {labels[0].shape}')
                #print(f'Train Output Shape: {output_reshaped.shape} \tLabels Shape: {labels[0].shape}')
                #print(output_reshaped.shape, labels[0])
                loss += focal_loss(output_reshaped, labels[0], num_class=10, alpha=-1, gamma=1)
                loss += tmse_loss(output_reshaped, labels[0], gamma=0.1)
            #for i, value in enumerate(outputs):
                #loss += focal_loss(value, labels[i], num_class=10, alpha=-1, gamma=1)
                #loss += tmse_loss(value, labels[i], gamma=0.1)

            loss = 10 * loss / splits
            loss.backward()
            optimizer.step()
            scheduler.step()
            print('Train Loss: ', loss.item())
            if math.isnan(loss.item()):
                raise Exception("Stopping the program due to an nan")

            train_loss.append(loss.item())
            train_prog += 1

            acc = 0
            for i, value in enumerate(outputs):
                #prob = F.softmax(value, dim=1)
                #pred = prob.data.max(dim=1)#[1]
                prob = F.softmax(value.permute(1, 0))
                pred = prob.data.max(dim=1)[1]

                # *************************************************************
                #modified print
                #print(f"pred: {pred}")
                #print(f"labels[{i}]: {labels[i]}")
                acc += torchmetrics.functional.accuracy(pred, labels[i], task="multiclass", num_classes=10).item() #.squeeze()
            print('Train acc / splits): ', acc / splits)
            train_acc.append(acc / splits)

            print(f"pred: {pred}")
            print(f"labels[{i}]: {labels[i]}")

        # Validation
        with torch.no_grad():
            net.eval()
           # print(f'-----------------------------Validation---------------')
            for iteration, (val_data, labels) in enumerate(val_dataloader):
                #print(f'+++++++++++++++++++++++++++++++Val Iteration: {iteration}++++++++++++++++++++++++++++++++++++++')
                #left_data = val_data[0].to(device)
                #right_data = val_data[1].to(device)
                #hands_data = torch.cat([left_data, right_data], dim=1)
                #labels = val_data[2].type(torch.long).to(device)
                val_data = val_data.to(device).permute(0,2,1)

                outputs = net(val_data, torch.ones(val_data.shape[0], 2, val_data.shape[2]).to(device))[-1]

                #outputs = net(val_data)[-1]
                labels = labels.to(device)

                loss_val = 0
                splits = labels.shape[0]
                #print('outputs.shape[0], labels[0].shape: ', outputs.shape[0], labels[0].shape)
                for i in range(outputs.shape[0]):
                    output_reshaped = outputs[i].squeeze()

                    #print(output_reshaped.shape, labels[0])
                    loss_val += focal_loss(output_reshaped, labels[0], num_class=10, alpha=-1, gamma=1)
                    loss_val += tmse_loss(output_reshaped, labels[0], gamma=0.1)
                #print('Val Loss: ', loss_val.item())
                loss_val = 10 * loss_val / splits
                print('Validation Loss: ', loss_val.item())
                val_loss.append(loss_val.item())
                val_prog += 1

                acc_val = 0
                for i in range(outputs.shape[0]):
                    output_reshaped = outputs[i].squeeze()
                    prob = F.softmax(output_reshaped, dim=1)
                    _, pred_indices = prob.data.max(dim=0)
                    acc_val += torchmetrics.functional.accuracy(pred, labels[i], task="multiclass", num_classes=10).item() #.squeeze()
                """for i, value in enumerate(outputs):
                    prob = F.softmax(value, dim=1)
                    pred = prob.data.max(dim=1)[1]
                    acc_val += torchmetrics.functional.accuracy(pred.squeeze(), labels.squeeze(), task="multiclass", num_classes=10).item()"""
                print('Val acc / splits): ', acc_val / splits)
                val_acc.append(acc_val / splits)
          
        # Logging and writing to CSV
        if ep % print_progress == 0 or ep == 1:
            curr_t_loss = sum(train_loss[-train_prog:]) / train_prog
            curr_t_acc = sum(train_acc[-train_prog:]) / train_prog
            curr_v_loss = sum(val_loss[-val_prog:]) / val_prog
            curr_v_acc = sum(val_acc[-val_prog:]) / val_prog

            writer.writerow([ep, curr_t_loss, curr_t_acc, curr_v_loss, curr_v_acc])

            print(f'[{ep}/{Epochs}]\tTraining Loss: {curr_t_loss:.4f} \tTraining Accuracy: {curr_t_acc:.4f} \tValidation Loss: {curr_v_loss:.4f} \tValidation Accuracy: {curr_v_acc:.4f}')
            train_prog = 0
            val_prog = 0

        if ep % 100 == 0:
            net_path = f'./models/ASformer_SlidingWindow_e{ep}.pt'
            torch.save(net.state_dict(), net_path)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Epochs = 1000
    Lr_Rate = 0.0001
    batch_size = 32

    data_dir = './data/train'
    train_dataset = SlidingWindowIMUsDataset(data_dir=data_dir, window_len=20000, hop=500, sample_len=2000,
                                             augmentation=False, ambidextrous=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dir = './data/validate/'
    val_dataset = SlidingWindowIMUsDataset(data_dir=val_dir, window_len=20000, hop=500, sample_len=2000,
                                           augmentation=False, ambidextrous=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # testing transformer
    net = ASformer(num_classes=10, r1=1, r2=1, num_decoders=2, num_f_maps=128, input_dim=44)
    #net = MS_TCN2(num_layers_PG=3, num_layers_R=3, num_R=4, num_f_maps=128, dim=44, num_classes=10)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=Lr_Rate, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Epochs//10, gamma=0.9)

    with open('models/results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy'])
        train(device, train_dataloader, val_dataloader, net, optimizer, scheduler, Epochs, 20, writer)

