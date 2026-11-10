import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import notebook
import itertools
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, multilabel_confusion_matrix
import time
import random
pd.options.mode.chained_assignment = None  # type: ignore # default='warn'
from torch.nn.functional import normalize
from sklearn import preprocessing
from sklearn import linear_model
from torch.utils.data import Dataset, DataLoader

########################################## Dataset class #######################################################################

class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

########################################## TRAIN FUNCTION for DAST #############################################################

def train_DAST(Xtrain, Ytrain, Xtest, Ytest, model, criterion, optimizer, scheduler, epochs, batch_size, info): # lst_filter
    t0 = time.time()
    # Boucle sur epoch pour entrainer le modèle
    df_epoch_loss = pd.DataFrame(columns = ["Epoch", "Loss_train", "Accuracy_train"]) # , "Loss_test", "Accuracy_test"  # type: ignore
    df_epoch_loss.Epoch = range(0, epochs)
    test_filter_loss = pd.DataFrame()
    Xtrain = torch.tensor(Xtrain[:, :, 0:-3], dtype=torch.float32)
    Ytrain = torch.tensor(Ytrain, dtype = torch.float32)
    Xtest = torch.tensor(Xtest[:, :, 0:-3], dtype=torch.float32)#.to(device)
    Ytest = torch.tensor(Ytest, dtype = torch.float32)#.to(device)
    traindata = MyDataset(Xtrain, Ytrain)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
    for j in range(epochs):
        loss, test_loss, sum_acc, num_acc, num_acc_test, acc_test = 0,0,0,0,0,list()
        print("Epoch", j)
        for i, data in enumerate(trainloader):
            loss_batch, num_acc_batch, acc_batch = 0, 0, list()
            xtrain, ytrain = data # get the inputs
            xtrain = xtrain#.to(device)
            ytrain = ytrain#.to(device)
            optimizer.zero_grad() # setting the gradient to 0
            y_pred = model.forward(xtrain) # forward pass
            for i in range(len(ytrain)):
                new_ytrain = ytrain[i][~torch.isnan(ytrain[i])]#.cpu()
                # y_pred = y_pred#.cpu()
                loss_batch += criterion(y_pred[i][0:new_ytrain.shape[0]], new_ytrain)
                ##### ACCURACY
                y_pred = y_pred.round()
                acc_batch.append(accuracy_score(new_ytrain.cpu().detach().numpy(), 
                                                y_pred[i][0:new_ytrain.shape[0]].cpu().detach().numpy(), normalize=False)) # .detach().numpy() .detach().numpy()
                num_acc_batch += len(new_ytrain)
            loss_batch.backward()
            optimizer.step()
            scheduler.step()
            sum_acc += np.sum(acc_batch)
            num_acc += num_acc_batch
            loss += loss_batch
        accuracy = sum_acc/num_acc
        ### Test results ###
        model.eval()
        with torch.no_grad():
            y_test_pred = model.forward(Xtest)
            y_test_pred = y_test_pred#.squeeze(0)
            for i in range(len(Ytest)):
                new_Ytest = Ytest[i][~torch.isnan(Ytest[i])]#.cpu()
                y_test_pred = y_test_pred#.cpu()
                test_loss += criterion(y_test_pred[i][0:new_Ytest.shape[0]], new_Ytest)
                ##### ACCURACY
                y_test_pred = y_test_pred.round()
                acc_test.append(accuracy_score(new_Ytest.detach().numpy(), y_test_pred[i][0:new_Ytest.shape[0]].detach().numpy(), normalize=False))
                num_acc_test += len(new_Ytest)
            sum_acc_test = np.sum(acc_test)                              
            accuracy_test = sum_acc_test/num_acc_test
        df_epoch_loss.loc[df_epoch_loss.Epoch == j, "Loss_train"] = loss.detach().numpy()
        df_epoch_loss.loc[df_epoch_loss.Epoch == j, "Accuracy_train"] = accuracy
        df_epoch_loss.loc[df_epoch_loss.Epoch == j, "Loss_test"] = test_loss.detach().numpy()
        df_epoch_loss.loc[df_epoch_loss.Epoch == j, "Accuracy_test"] = accuracy_test
        print("Epoch loss :", "%.4f" % loss, "Accuracy :", "%.4f" % accuracy, "Test loss :" "%.4f" % test_loss, "Accuracy test :", "%.4f" % accuracy_test)
        if test_loss <= np.min(df_epoch_loss.Loss_test):
            torch.save({
            'epoch': j,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_sate_dict': scheduler.state_dict(),
            # 'test_loss': test_loss.item(),
            # 'test_accuracy': accuracy_test,
            'train_loss' : loss,
            'train_accuracy' : accuracy,
            }, "./model_checkpoint/best_model_test_"+str(info)+".pkl")
        if loss <= np.min(df_epoch_loss.Loss_train):
            torch.save({
            'epoch': j,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_sate_dict': scheduler.state_dict(),
            # 'test_loss': test_loss.item(),
            # 'test_accuracy': accuracy_test,
            'train_loss' : loss.cpu().detach().numpy(),
            'train_accuracy' : accuracy,
            }, "./model_checkpoint/best_model_train_"+str(info)+".pkl")
        if (j % 100) == 0:
            torch.save({
            'epoch': j,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_sate_dict': scheduler.state_dict(),
            # 'test_loss': test_loss.item(),
            # 'test_accuracy': accuracy_test,
            'train_loss' : loss.cpu().detach().numpy(),
            'train_accuracy' : accuracy,
            }, "./model_checkpoint/model_epoch_"+str(j)+"_"+str(info)+".pkl")
            print(str(info))
    df_epoch_loss.to_csv("./model_checkpoint/results/df_epoch_loss_"+str(info)+".csv")
    t1 = time.time() - t0
    print(t1)
    return()