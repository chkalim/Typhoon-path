

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from geopy.distance import great_circle

from sklearn import svm
from sklearn.metrics import accuracy_score
import sklearn.metrics

import torch
import torch.utils.data as Data
import pickle
import datetime

import os
import random

# forecast 24-hour lead time
pre_seq = 4
batch_size = 128
epochs = 128
min_val_loss = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

import math

train = pd.read_csv('data/CMA_train_' + str( pre_seq* 6) + 'h.csv',
                    header=None)
test = pd.read_csv('data/CMA_test_' + str(pre_seq * 6) + 'h.csv',
                   header=None)

train.shape, test.shape


class STLSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(STLSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)

        self.lstm4 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 53)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, test=0):
        x = x.unsqueeze(1)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        h_2 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_2 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        h_3 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_3 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        # out = self.fc1(h_out)
        out = h_out.unsqueeze(1)

        ula2, (h_out2, _) = self.lstm2(out, (h_1, c_1))
        h_out2 = h_out2.view(-1, self.hidden_size)

        out2 = h_out2.unsqueeze(1)
        ula3, (h_out3, _) = self.lstm3(out2, (h_2, c_2))
        h_out3 = h_out3.view(-1, self.hidden_size)
        # out3 = self.fc(h_out3)

        out3 = h_out3.unsqueeze(1)
        ula4, (h_out4, _) = self.lstm4(out3, (h_3, c_3))
        h_out4 = h_out4.view(-1, self.hidden_size)
        out4 = self.fc(h_out4)


        return out4


class STGRU(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(STGRU, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.gru3 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, test=0):
        x = x.unsqueeze(1)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        h_2 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        ula, _ = self.gru(x, h_0)

        # h_out = ula.view(-1, self.hidden_size)
        #
        # h_out = h_out.unsqueeze(1)
        ula2, _ = self.gru2(ula, h_1)
        ula3, _ = self.gru3(ula2, h_2)

        h_out = ula3.view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out


class TrainLoader(Data.Dataset):
    def __init__(self, X_wide_train, y_train):
        self.X_wide_train = X_wide_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.X_wide_train[index], self.y_train[index]

    def __len__(self):
        return len(self.X_wide_train)


CLIPER_feature = pd.concat((train, test), axis=0)
CLIPER_feature.reset_index(drop=True, inplace=True)

X_wide_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
y1_scaler = MinMaxScaler()
y2_scaler = MinMaxScaler()
X_wide = X_wide_scaler.fit_transform(CLIPER_feature.iloc[:, 6:])

X_wide_train = X_wide[0: train.shape[0], :]


y = y_scaler.fit_transform(CLIPER_feature.loc[:, 3:4])
# y1=CLIPER_feature.loc[:, 3]
# y2=CLIPER_feature.loc[:, 4]
# y1 = y1_scaler.fit_transform(y1)
y1 = y1_scaler.fit_transform(CLIPER_feature.loc[:, 3:3])
y2 = y2_scaler.fit_transform(CLIPER_feature.loc[:, 4:4])


# ****** modified


y_train = y1[0: train.shape[0], :]

# train_index, meta_train_index, _, _, = train_test_split(full_train_index, full_train_index, test_size=0.30)
#
# meta_train_index=X_wide_train[meta_train_index]
# meta_y_train=y_train[meta_train_index]


print("************************************************")
print(len(X_wide_train))
print(len(y_train))

# ********** here is the code for extracting test code
# X_wide_test = X_wide[train.shape[0]: , :]
# y_test = y[train.shape[0]: , :]

# print(len(X_wide_test))
# print(len(y_test))


learning_rate = 0.01

input_size = 53
hidden_size = 128
num_layers = 1

num_classes = 2



model_STLSTM1 = STLSTM(num_classes, input_size, hidden_size, num_layers)
model_STLSTM2 = STLSTM(num_classes, input_size, hidden_size, num_layers)
model_STLSTM3 = STLSTM(num_classes, input_size, hidden_size, num_layers)
model_STLSTM4 = STLSTM(num_classes, input_size, hidden_size, num_layers)
model_STLSTM5 = STLSTM(num_classes, input_size, hidden_size, num_layers)
model_STLSTM6 = STLSTM(num_classes, input_size, hidden_size, num_layers)
model_STLSTM7 = STLSTM(num_classes, input_size, hidden_size, num_layers)
model_STLSTM8 = STLSTM(num_classes, input_size, hidden_size, num_layers)
model_STLSTM9 = STLSTM(num_classes, input_size, hidden_size, num_layers)
model_STLSTM10 = STLSTM(num_classes, input_size, hidden_size, num_layers)


model_STGRU1 = STGRU(num_classes, input_size, hidden_size, num_layers)
model_STGRU2 = STGRU(num_classes, input_size, hidden_size, num_layers)
model_STGRU3 = STGRU(num_classes, input_size, hidden_size, num_layers)
model_STGRU4 = STGRU(num_classes, input_size, hidden_size, num_layers)
model_STGRU5 = STGRU(num_classes, input_size, hidden_size, num_layers)
model_STGRU6 = STGRU(num_classes, input_size, hidden_size, num_layers)
model_STGRU7 = STGRU(num_classes, input_size, hidden_size, num_layers)
model_STGRU8 = STGRU(num_classes, input_size, hidden_size, num_layers)
model_STGRU9 = STGRU(num_classes, input_size, hidden_size, num_layers)
model_STGRU10 = STGRU(num_classes, input_size, hidden_size, num_layers)


years = test[5].unique()
test_list = []

for year in years:
    temp = test[test[5] == year]
    temp = temp.reset_index(drop=True)
    test_list.append(temp)


# temp = test[test[1] == 2350]
# temp = temp.reset_index(drop=True)
# test_list.append(temp)

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = model
# net = net.to(device)


model_name_STLSTM1 = 'model_save_ens/model_multilayer/Model_STlstm1.pkl'
model_name_STLSTM2 = 'model_save_ens/model_multilayer/Model_STlstm2.pkl'
model_name_STLSTM3 = 'model_save_ens/model_multilayer/Model_STlstm3.pkl'
model_name_STLSTM4 = 'model_save_ens/model_multilayer/Model_STlstm4.pkl'
model_name_STLSTM5 = 'model_save_ens/model_multilayer/Model_STlstm5.pkl'
model_name_STLSTM6 = 'model_save_ens/model_multilayer/Model_STlstm6.pkl'
model_name_STLSTM7 = 'model_save_ens/model_multilayer/Model_STlstm7.pkl'
model_name_STLSTM8 = 'model_save_ens/model_multilayer/Model_STlstm8.pkl'
model_name_STLSTM9 = 'model_save_ens/model_multilayer/Model_STlstm9.pkl'
model_name_STLSTM10 = 'model_save_ens/model_multilayer/Model_STlstm10.pkl'

model_name_STGRU1 = 'model_save_ens/model_multilayer/Model_STgru1.pkl'
model_name_STGRU2 = 'model_save_ens/model_multilayer/Model_STgru2.pkl'
model_name_STGRU3 = 'model_save_ens/model_multilayer/Model_STgru3.pkl'
model_name_STGRU4 = 'model_save_ens/model_multilayer/Model_STgru4.pkl'
model_name_STGRU5 = 'model_save_ens/model_multilayer/Model_STgru5.pkl'
model_name_STGRU6 = 'model_save_ens/model_multilayer/Model_STgru6.pkl'
model_name_STGRU7 = 'model_save_ens/model_multilayer/Model_STgru7.pkl'
model_name_STGRU8 = 'model_save_ens/model_multilayer/Model_STgru8.pkl'
model_name_STGRU9 = 'model_save_ens/model_multilayer/Model_STgru9.pkl'
model_name_STGRU10 = 'model_save_ens/model_multilayer/Model_STgru10.pkl'




model_STLSTM1.load_state_dict(torch.load(model_name_STLSTM1))
model_STLSTM2.load_state_dict(torch.load(model_name_STLSTM2))
model_STLSTM3.load_state_dict(torch.load(model_name_STLSTM3))
model_STLSTM4.load_state_dict(torch.load(model_name_STLSTM4))
model_STLSTM5.load_state_dict(torch.load(model_name_STLSTM5))
model_STLSTM6.load_state_dict(torch.load(model_name_STLSTM6))
model_STLSTM7.load_state_dict(torch.load(model_name_STLSTM7))
model_STLSTM8.load_state_dict(torch.load(model_name_STLSTM8))
model_STLSTM9.load_state_dict(torch.load(model_name_STLSTM9))
model_STLSTM10.load_state_dict(torch.load(model_name_STLSTM10))


model_STGRU1.load_state_dict(torch.load(model_name_STGRU1))
model_STGRU2.load_state_dict(torch.load(model_name_STGRU2))
model_STGRU3.load_state_dict(torch.load(model_name_STGRU3))
model_STGRU4.load_state_dict(torch.load(model_name_STGRU4))
model_STGRU5.load_state_dict(torch.load(model_name_STGRU5))
model_STGRU6.load_state_dict(torch.load(model_name_STGRU6))
model_STGRU7.load_state_dict(torch.load(model_name_STGRU7))
model_STGRU8.load_state_dict(torch.load(model_name_STGRU8))
model_STGRU9.load_state_dict(torch.load(model_name_STGRU9))
model_STGRU10.load_state_dict(torch.load(model_name_STGRU10))



total_seq=10
# model.eval()
# models = [model1, model2, model3]

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

def stacked_dataset(members,inputx):
    stackx=None
    for model in members:
        pred_value= model(inputx)
        #pred_value=y_scaler.inverse_transform(pred_value.detach().numpy())
        pred_value = pred_value.detach().numpy()
        if stackx is None:
            stackx = pred_value
        else:
            stackx = np.dstack((stackx, pred_value))
            #stackx = np.append(stackx, pred_value, axis=1)
    stackx = stackx.reshape((stackx.shape[0], stackx.shape[1] * stackx.shape[2]))
    return stackx
from sklearn import datasets, linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn import neighbors
# fit a model based on the outputs from the ensemble members

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.naive_bayes import GaussianNB,MultinomialNB


def fit_stacked_model(members, inputX, inputy1,inputy2):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
   
    input_1 = stackedX[:, :total_seq]
    input_2 = stackedX[:, total_seq:]
   
    model1 = LinearRegression()
    model2 = LinearRegression()
    

    model1.fit(input_1, inputy1.ravel())
    model2.fit(input_2, inputy2.ravel())
    return model1,model2

def stacked_prediction(members, model1,model2, inputX):
    stackedX = stacked_dataset(members, inputX)
    input_1 = stackedX[:, :total_seq]
    input_2 = stackedX[:, total_seq:]

    
    yhat1 = model1.predict(input_1)
    yhat2 = model2.predict(input_2)
    pred=np.vstack((yhat1, yhat2))
    #pred = np.append(yhat1, yhat2,axis=0)
    pred=(np.transpose(pred))
    return pred

members=[model_STLSTM1,model_STLSTM2,model_STLSTM3,model_STLSTM4,model_STLSTM5,model_STLSTM6
        ,model_STLSTM7,model_STLSTM8,model_STLSTM9,model_STLSTM10,model_STGRU1,model_STGRU2,model_STGRU3,model_STGRU4
        ,model_STGRU5,model_STGRU6,model_STGRU7,model_STGRU8,model_STGRU9,model_STGRU10]

count_index=0
limit=int((len(X_wide_train)/100)*50)
# X_wide_train_a = X_wide[0: limit, :]
X_meta_train = X_wide[limit: train.shape[0], :]
# y_wide_train_a = y[0: limit, :]
y1_meta_train = y1[limit: train.shape[0], :]
y2_meta_train = y2[limit: train.shape[0], :]

X_meta_train = Variable(torch.from_numpy(X_meta_train).float())
model1,model2 = fit_stacked_model(members,X_meta_train , y1_meta_train,y2_meta_train)

with torch.no_grad():
    for year, _test in zip(years, test_list):
        print(year, 'Ã¥Â¹Â´:')

        y_test_lat = _test.loc[:, 3]

        y_test_long = _test.loc[:, 4]

        X_wide_test = X_wide_scaler.transform(_test.loc[:, 6:])
        #X_wide_test = _test.loc[:, 6:]

        final_test_list = []

        X_wide_test = Variable(torch.from_numpy(X_wide_test).float())
 

        pred1 = stacked_prediction(members, model1,model2, X_wide_test)
        pred = y_scaler.inverse_transform(pred1)
     

        pred_lat = pred[:, 0]
        pred_long = pred[:, 1]

        true_lat = y_test_lat
        true_long = y_test_long

        diff_lat = np.abs(pred_lat - true_lat)
        diff_long = np.abs(pred_long - true_long)

        print('avg lat:', sum(diff_lat) / len(diff_lat))
        print('avg long:', sum(diff_long) / len(diff_long))

        sum_error = []
        for i in range(0, len(pred_lat)):
            sum_error.append(great_circle((pred_lat[i], pred_long[i]), (true_lat[i], true_long[i])).kilometers)

        print('avg distance error:', sum(sum_error) / len(sum_error))
