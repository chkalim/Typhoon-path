# -*- coding: utf-8 -*-
"""CMA_preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_85d3nSnHUcYNgWNu2i79cY43P6uW1iu

# Definition
"""


forward_seq = 4
backward_seq = 4
valid_tropical_seq_num =4
predict_seq_num = 4
trainYear = (1949, 2014)
testYear = (2015, 2018)

"""
typhoonYear
tid Typhoon ID
typhoonRecordNum 
typhoonName 
typhoonRecords 
"""
class TyphoonHeader:

    def __init__(self, typhoonYear, tid):
        self.typhoonYear = typhoonYear
        self.tid = tid
        self.typhoonRecordNum = 0
        self.typhoonRecords = []

    def printer(self):
        print("tid: %d, typhoonYear: %d, typhoonYear: %s, typhoonRecordNum: %d" %
              (self.tid, self.typhoonYear, self.typhoonName, self.typhoonRecordNum))
        for typhoonRecord in self.typhoonRecords:
            typhoonRecord.printer()


class TyphoonRecord:

    def __init__(self, typhoonTime, lat, long, wind, totalNum):
        self.typhoonTime = typhoonTime
        self.lat = lat
        self.long = long
        self.wind = wind
        self.totalNum = totalNum

    def printer(self):
        print("totalNum: %d, typhoonTime: %d, lat: %d, long: %d, wind: %d" %
              (self.totalNum, self.typhoonTime, self.lat, self.long, self.wind))

import math
import numpy as np

def get_vector_angle(vector1, vector2):
    a_abs = math.sqrt(vector1[0]*vector1[0] + vector1[1] * vector1[1])
    b_abs = math.sqrt(vector2[0]*vector2[0] + vector2[1] * vector2[1])
    a_b = vector1[0]*vector2[0] + vector1[1]*vector2[1]
    cos_angle = a_b/(a_abs*b_abs)
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1
    angle = np.arccos(cos_angle) * 360 / 2 / np.pi
    return angle

def read_cma_dfs(cma_list):
    totalNum = 1
    typhoonHeaderList = []
    for df in cma_list:
        tid = df.loc[0, 'TID']
        typhoonYear = df.loc[0, 'YEAR']
        typhoonHeader = TyphoonHeader(typhoonYear, tid, )
        for i in range(len(df)):
            typhoonTime = int(
                str(df.loc[i, 'YEAR'])+
                str(df.loc[i, 'MONTH']).zfill(2)+
                str(df.loc[i, 'DAY']).zfill(2)+
                str(df.loc[i, 'HOUR']).zfill(2)
            )
            lat = df.loc[i, 'LAT'] * 0.1
            long = df.loc[i, 'LONG'] * 0.1
            wind = df.loc[i, 'WND']
            typhoonRecord = TyphoonRecord(typhoonTime, lat, long, wind, totalNum)
            totalNum += 1
            typhoonHeader.typhoonRecords.append(typhoonRecord)
        typhoonHeader.typhoonRecordNum = len(typhoonHeader.typhoonRecords)
        typhoonHeaderList.append(typhoonHeader)
    return typhoonHeaderList

from geopy.distance import great_circle

def buildup_feature(typhoonRecordsList, tid, fileXWriter, pred_num):
    for start in range(forward_seq, len(typhoonRecordsList) - predict_seq_num):
        strXLine = str(typhoonRecordsList[start].totalNum) + \
                   ',' + str(tid) + \
                   ',' + str(typhoonRecordsList[start].typhoonTime) + \
                   ',' + str(typhoonRecordsList[start + pred_num].lat) + \
                   ',' + str(typhoonRecordsList[start + pred_num].long) + \
                   ',' + str(typhoonRecordsList[start].typhoonTime//1000000) + \
                   ',' + str(typhoonRecordsList[start].lat) + \
                   ',' + str(typhoonRecordsList[start].long) + \
                   ',' + str(typhoonRecordsList[start].wind)

        # ???24?????????????????????
        strXLine += ',' + str(typhoonRecordsList[start - 1].lat)
        strXLine += ',' + str(typhoonRecordsList[start - 2].lat)
        strXLine += ',' + str(typhoonRecordsList[start - 3].lat)
        strXLine += ',' + str(typhoonRecordsList[start - 4].lat)
        
        # ???24?????????????????????
        strXLine += ',' + str(typhoonRecordsList[start - 1].long)
        strXLine += ',' + str(typhoonRecordsList[start - 2].long)
        strXLine += ',' + str(typhoonRecordsList[start - 3].long)
        strXLine += ',' + str(typhoonRecordsList[start - 4].long)
        
        # ???24?????????????????????
        strXLine += ',' + str(typhoonRecordsList[start - 1].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 2].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 3].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 4].wind)

        # ????????????
        strXLine += ',' + str(typhoonRecordsList[start].typhoonTime//10000 % 100)
        
        # ?????????
        strXLine += ',' + str(typhoonRecordsList[start].wind - typhoonRecordsList[start - 1].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 1].wind - typhoonRecordsList[start - 2].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 2].wind - typhoonRecordsList[start - 3].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 3].wind - typhoonRecordsList[start - 4].wind)

        # ????????? 0-6, 0-12,0-18,0-24
        latdiff = []
        latdiff.append((typhoonRecordsList[start].lat - typhoonRecordsList[start - 1].lat))
        strXLine += ',' + str(latdiff[-1])
        latdiff.append((typhoonRecordsList[start - 1].lat - typhoonRecordsList[start - 2].lat))
        strXLine += ',' + str(latdiff[-1])
        latdiff.append((typhoonRecordsList[start - 2].lat - typhoonRecordsList[start - 3].lat))
        strXLine += ',' + str(latdiff[-1])
        latdiff.append((typhoonRecordsList[start - 3].lat - typhoonRecordsList[start - 4].lat))
        strXLine += ',' + str(latdiff[-1])

        # ?????????
        longdiff = []
        longdiff.append((typhoonRecordsList[start].long - typhoonRecordsList[start - 1].long))
        strXLine += ',' + str(longdiff[-1])
        longdiff.append((typhoonRecordsList[start - 1].long - typhoonRecordsList[start - 2].long))
        strXLine += ',' + str(longdiff[-1])
        longdiff.append((typhoonRecordsList[start - 2].long - typhoonRecordsList[start - 3].long))
        strXLine += ',' + str(longdiff[-1])
        longdiff.append((typhoonRecordsList[start - 3].long - typhoonRecordsList[start - 4].long))
        strXLine += ',' + str(longdiff[-1])

        # ??????????????????????????????
        sum = 0
        for i in range(len(latdiff)):
            sum += latdiff[i]**2
        strXLine += ',' + str(sum)
        strXLine += ',' + str("{:.4f}".format(math.sqrt(sum)))

        # ??????????????????????????????
        sum = 0
        for i in range(len(latdiff)):
            sum += longdiff[i] ** 2
        strXLine += ',' + str(sum)
        strXLine += ',' + str("{:.4f}".format(math.sqrt(sum)))

        # ???????????????
        strXLine += ',' + str("{:.4f}".format(
            2 * great_circle(
                (typhoonRecordsList[start].lat, typhoonRecordsList[start].long),
                (typhoonRecordsList[start - 1].lat,typhoonRecordsList[start - 1].long)
            ).kilometers / (6 ** 2)
        ))
        strXLine += ',' + str("{:.4f}".format(
            2 * great_circle(
                (typhoonRecordsList[start - 1].lat,typhoonRecordsList[start - 1].long),
                (typhoonRecordsList[start - 2].lat,typhoonRecordsList[start - 2].long)
            ).kilometers / (6 ** 2)
        ))
        strXLine += ',' + str("{:.4f}".format(
            2 * great_circle(
                (typhoonRecordsList[start].lat,typhoonRecordsList[start].long),
                (typhoonRecordsList[start - 2].lat,typhoonRecordsList[start - 2].long)
            ).kilometers / (12 ** 2)
                                            ))
        strXLine += ',' + str("{:.4f}".format(
            2 * great_circle(
                (typhoonRecordsList[start -2].lat,typhoonRecordsList[start -2].long),
                (typhoonRecordsList[start - 4].lat,typhoonRecordsList[start - 4].long)
            ).kilometers / (12 ** 2)
        ))

        # ?????????????????????
        strXLine += ',' + str("{:.4f}".format(math.sqrt(typhoonRecordsList[start].lat)))

        # ?????????????????????
        strXLine += ',' + str("{:.4f}".format(math.sqrt(typhoonRecordsList[start].long)))

        ########################################################################################################

        # ???????????????
        strXLine += ',' + str(
            (typhoonRecordsList[start].lat - typhoonRecordsList[start - 1].lat) -
            (typhoonRecordsList[start - 1].lat - typhoonRecordsList[start - 2].lat)
        )

        strXLine += ',' + str(
            (typhoonRecordsList[start - 2].lat - typhoonRecordsList[start - 3].lat) -
            (typhoonRecordsList[start - 3].lat - typhoonRecordsList[start - 4].lat)
        )
        
        # ???????????????
        strXLine += ',' + str(
            (typhoonRecordsList[start].long - typhoonRecordsList[start - 1].long) -
            (typhoonRecordsList[start - 1].long - typhoonRecordsList[start - 2].long)
        )

        strXLine += ',' + str(
            (typhoonRecordsList[start - 2].long - typhoonRecordsList[start - 3].long) -
            (typhoonRecordsList[start - 3].long - typhoonRecordsList[start - 4].long)
        )
        
        # ??????????????????
        for i in range(1, forward_seq + 1):
            diff_lat = typhoonRecordsList[start - i + 1].lat - typhoonRecordsList[start - i].lat
            diff_long = typhoonRecordsList[start - i + 1].long - typhoonRecordsList[start - i].long
            vector1 = [diff_lat, diff_long]
            vector2 = [1, 0]
            if diff_lat < 0 and diff_long < 0:
                strXLine += ',' + str(90 + get_vector_angle(vector1, vector2))
            elif diff_lat > 0 and diff_long < 0:
                strXLine += ',' + str(270 + get_vector_angle(vector1, vector2))
            elif diff_lat == 0 and diff_long == 0:
                strXLine += ',' + str(0)
            else:
                strXLine += ',' + str(get_vector_angle(vector1, vector2))

        # ?????????????????????
        for i in range(1, forward_seq + 1):
            diff_lat = typhoonRecordsList[start - i + 1].lat - typhoonRecordsList[start - i].lat
            diff_long = typhoonRecordsList[start - i + 1].long - typhoonRecordsList[start - i].long
            vector1 = [diff_lat, diff_long]
            vector2 = [0, 1]
            if diff_lat > 0 and diff_long < 0:
                strXLine += ',' + str(90 + get_vector_angle(vector1, vector2))
            elif diff_lat > 0 and diff_long > 0:
                strXLine += ',' + str(270 + get_vector_angle(vector1, vector2))
            elif diff_lat == 0 and diff_long == 0:
                strXLine += ',' + str(0)
            else:
                strXLine += ',' + str(get_vector_angle(vector1, vector2))
        
        # ???????????????????????????????????????
        for i in range(1, forward_seq):
            diff_lat1 = typhoonRecordsList[start - i + 1].lat - typhoonRecordsList[start - i].lat
            diff_long1 = typhoonRecordsList[start - i + 1].long - typhoonRecordsList[start - i].long
            vector1 = [diff_lat1, diff_long1]
            diff_lat2 = typhoonRecordsList[start - i - 1].lat - typhoonRecordsList[start - i].lat
            diff_long2 = typhoonRecordsList[start - i - 1].long - typhoonRecordsList[start - i].long
            vector2 = [diff_lat2, diff_long2]
            if diff_lat1 == 0 and diff_long1 == 0 or diff_lat2 == 0 and diff_long2 == 0:
                strXLine += ',' + str(0)
            else:
                strXLine += ',' + str(get_vector_angle(vector1, vector2))

        fileXWriter.write(strXLine + '\n')

"""# Read CMA"""

import os
import pandas as pd

path = "data/CMA"
files = os.listdir(path)
files.sort()

pd_list = []
for file in files:
    cma_pd = pd.read_csv(path+'//'+file, delim_whitespace=True, 
                         names=['TROPICALTIME', 'I', 'LAT', 'LONG', 'PRES', 'WND' , 'OWD', 'NAME', 'RECORDTIME'])
    pd_list.append(cma_pd)

df = pd.concat(pd_list, axis=0)

df=df.reset_index(drop=True)

df

"""# CMA data"""

df = df.drop(columns=['OWD','RECORDTIME'])

df = pd.concat([df, pd.DataFrame(columns=['TID','YEAR','MONTH','DAY','HOUR'])], axis=1)

df = df[['TID','YEAR','MONTH','DAY','HOUR','TROPICALTIME', 'I', 'LAT', 'LONG', 'WND', 'PRES', 'NAME']]


tid = 0
name = None
for i in range(0, len(df)):
    if df.at[i, 'TROPICALTIME'] == 66666:
        tid += 1
        name = df.loc[i, 'NAME']
    else:
        df.at[i, 'TID'] = tid
        df.at[i, 'NAME'] = name
        df.at[i, 'YEAR'] = df.loc[i, 'TROPICALTIME'] // 1000000
        df.at[i, 'MONTH'] = df.loc[i, 'TROPICALTIME'] // 10000 % 100
        df.at[i, 'DAY'] = df.loc[i, 'TROPICALTIME'] // 100 % 100
        df.at[i, 'HOUR'] = df.loc[i, 'TROPICALTIME'] % 100

df = df.drop(df[df['TROPICALTIME']==66666].index, axis=0)
df = df.drop(columns=['TROPICALTIME'])

df=df.reset_index(drop=True)

df

df.loc[df['NAME']=='In-fa', 'NAME'] = 'Infa'

df[df['NAME']=='Infa']


# ??????KEY???
df['KEY'] = None
# ???????????????????????????
years = df['YEAR'].unique()
years_dict = dict(zip(years , np.ones(years.shape)))


# ????????????????????????
result_list = []
# ????????????????????????????????????
for tid in df['TID'].unique():
    temp_df = df[df['TID']==tid].copy()
    # ?????????????????????????????????
    tid_year = temp_df['YEAR'].unique()[0]
    cy = int(years_dict[tid_year])
    years_dict[tid_year] += 1
    temp_df['KEY'] = str(tid_year) + '-' + str(cy).zfill(2)
    result_list.append(temp_df)
    
df = pd.concat(result_list, axis=0)

# ?????????????????????
df=df.reset_index(drop=True)

df


"""# Unique"""

import pandas as pd

df = df.drop(df[~df['HOUR'].isin([0,6,12,18])].index, axis=0)
df=df.reset_index(drop=True)

df

df = df.drop_duplicates()
df=df.reset_index(drop=True)

df


df.to_csv('data/raw.csv', index=False)

print('raw.csv ????????????')

"""# Data Preprocessing"""

import pandas as pd

# ??????????????????????????????dataframe
df = pd.read_csv('data/raw.csv')

tids = df['TID'].unique()

cma_list = []
cma_list = []

for tid in tids:
    temp_df = df[df['TID']==tid]
    temp_df = temp_df.reset_index(drop=True)
    cma_list.append(temp_df)
len(cma_list)

valid_tropical_len = forward_seq + valid_tropical_seq_num + backward_seq
print(len(cma_list))
temp = []
for df in cma_list:
    if df.shape[0] >= valid_tropical_len:
        temp.append(df)

cma_list = temp


df = pd.concat(cma_list, axis=0)

df=df.reset_index(drop=True)

train_range = [ str(x) for x in range(trainYear[0], trainYear[1]+1) ]
train_keys =  [v for i, v in enumerate(df['KEY'].unique()) if any(s in v for s in train_range)]

test_range = [ str(x) for x in range(testYear[0], testYear[1]+1) ]
test_keys =  [v for i, v in enumerate(df['KEY'].unique()) if any(s in v for s in test_range)]

df = df[(df['KEY'].isin(train_keys)) | (df['KEY'].isin(test_keys))]
df=df.reset_index(drop=True)

tname = pd.read_csv('data/typhoon_name.csv')

dict_name = {}
for i in range(len(tname)):
    dict_name[tname.at[i, 'en'].lower()] = tname.at[i, 'cn']
dict_name['(nameless)']='?????????'

df['CN_NAME'] = None
for i in range(len(df)):
    try:
        df.at[i, 'CN_NAME'] = dict_name[df.at[i, 'NAME'].lower()]
    except KeyError:
        print(df.at[i, 'NAME'].lower())

df.to_csv('data/pre_processing.csv', index=False)

df

typhoonHeaderList = read_cma_dfs(cma_list)

for i in range(1, predict_seq_num+1):
    trainXFile = open('data/CMA_train_'+str(i*6)+'h.csv', 'w')
    testXFile = open('data/CMA_test_'+str(i*6)+'h.csv', 'w')
    for typhoonHeader in typhoonHeaderList:
        typhoonRecordsList = typhoonHeader.typhoonRecords
        if typhoonHeader.typhoonYear in range(trainYear[0], trainYear[1]+1):
            buildup_feature(typhoonRecordsList, typhoonHeader.tid, trainXFile, i)
        elif typhoonHeader.typhoonYear in range(testYear[0], testYear[1]+1):
            buildup_feature(typhoonRecordsList, typhoonHeader.tid, testXFile, i)

    trainXFile.close()
    testXFile.close()

"""# For Reanalysis data"""

cma_ecwmf_train = open('data/cma_ecwmf_train.csv', 'w')
cma_ecwmf_test = open('data/cma_ecwmf_test.csv', 'w')

for typhoonHeader in typhoonHeaderList:
    typhoonRecordsList = typhoonHeader.typhoonRecords
    for start in range(0, len(typhoonRecordsList) - 4):
        strXLine = str(typhoonRecordsList[start].totalNum) + \
           ',' + str(typhoonRecordsList[start].typhoonTime) + \
           ',' + str(typhoonRecordsList[start].lat) + \
           ',' + str(typhoonRecordsList[start].long) + \
           ',' + str(typhoonHeader.tid)
        if typhoonHeader.typhoonYear in range(trainYear[0], trainYear[1]+1):
            cma_ecwmf_train.write(strXLine + '\n')
        elif typhoonHeader.typhoonYear in range(testYear[0], testYear[1]+1):
            cma_ecwmf_test.write(strXLine + '\n')

cma_ecwmf_train.close()
cma_ecwmf_test.close()

