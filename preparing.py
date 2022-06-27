#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:57:37 2022

@author: Thomas_yanghan
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
# Random Forest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# xgboost
from xgboost import XGBClassifier


df = pd.read_excel("CAC_2022_Training+Data+Set+New.xlsx")
df_state = pd.read_excel("American_states.xlsx")

COLS = ['sor', 'cdf_seq_no', 'trans_desc', 'merchant_cat_code', 'amt',
       'db_cr_cd', 'payment_reporting_category', 'payment_category',
       'is_international', 'default_brand', 'default_location', 'qrated_brand',
       'coalesced_brand', 'Category']

## Preprocessing
### Dealing with default location, to state
dloc = df['default_location'].apply(lambda x: x.split(" ")[-1] if pd.notnull(x) else "NA")
dloc1 = dloc.apply(lambda x: x if len(x)==2 else "NA")
df['state'] = dloc1
states = list(df_state['shortname'])
dloc2 = dloc.apply(lambda x: 1 if x in states else 0)
df['isInUSA'] = dloc2
b2text = {0:"not in America", 1:"in America"}
dloc2_1 = dloc2.apply(lambda x: b2text[x])
df['isInUSA_1'] = dloc2_1
short2full = dict(zip(df_state['shortname'], df_state['fullname']))
dloc3 = dloc1.apply(lambda x: short2full[x] if x in short2full.keys() else "NA")
df['state_fullname'] = dloc3
dloc_city = df['default_location'].apply(lambda x: ''.join(x.split(" ")[0:-1]) if pd.notnull(x) else "NA")
dloc_city1= dloc_city.apply(lambda x: x if "111" not in x else "")
df['city'] = dloc_city1
detailed_location = ['' for i in range(len(df))]
for idx,state in enumerate(df['state_fullname']):
    detailed_location[idx] = df['city'][idx]+' ,'+state+', '+df['isInUSA_1'][idx]
df['detailed_location']=detailed_location


# Merchant_cat_code
# https://classification.codes/classifications/industry/mcc/
mcc = df['merchant_cat_code'].fillna(-1)
mcc_1 = ['' for i in range(len(mcc))]
for idx,m in enumerate(mcc):
    if m<=99:
        mcc_1[idx] = "NA"
    elif m<=699:
        mcc_1[idx] = "Reserved"
    elif m<=999:
        mcc_1[idx] = "Agricultural services"
    elif m<=1499:
        mcc_1[idx] = "Reserved"
    elif m<=2999:
        mcc_1[idx] = "Contracted services"
    elif m<=3999:
        mcc_1[idx] = "Reserved for private use"
    elif m<=4799:
        mcc_1[idx] = "Transportation"
    elif m<=4999:
        mcc_1[idx] = "Utilities"
    elif m<=5500:
        mcc_1[idx] = "Retail outlets-Non Automobiles and vehicles"
    elif m<=5599:
        mcc_1[idx] = "Retail outlets-Automobiles and vehicles"
    elif m<=5699:
        mcc_1[idx] = "Clothing outlets"
    elif m<=5999:
        mcc_1[idx] = "Miscellaneous outlets"
    elif m<=7299:
        mcc_1[idx] = "Service providers"
    elif m<=7529:
        mcc_1[idx] = "Business services"
    elif m<=7799:
        mcc_1[idx] = "Repair services"
    elif m<=7999:
        mcc_1[idx] = "Amusement and entertainment"
    elif m<=8999:
        mcc_1[idx] = "Professional services and membership organizations"
    elif m<=9199:
        mcc_1[idx] = "Reserved for ISO use"
    elif m<=9402:
        mcc_1[idx] = "Government services"
    elif m<=9999:
        mcc_1[idx] = "Reserved"
df['merchant_cat'] = mcc_1
        
        
### Labelization
df['db_cr_cd'] = df['db_cr_cd'].fillna('NA')
cols = []
for idx,c in enumerate(['sor', 'db_cr_cd', 'is_international', 'payment_category', 'state', 'merchant_cat']):
    temp_vs = list(set(df[c]))
    temp_l = [i for i in range(len(temp_vs))]
    hotdict = dict(zip(temp_vs, temp_l))
    df[c+'_1'] = df[c].apply(lambda x: hotdict[x])
    cols.append(c+'_1')


cates = list(set(df['Category']))
ns = [i for i in range(len(cates))]
hotdict = dict(zip(cates, ns))
df['Category_1'] = df['Category'].apply(lambda x: hotdict[x])



df_raw = df.copy()
df_raw.to_csv("train_processed.csv", index=False)
df = df_raw.copy()
df_1 = df[df['merchant_cat']!='NA']



X1 = df[['amt'] + cols]
# X2 = df[['sor', 'cdf_seq_no', 'trans_desc', 'merchant_cat_code', 'amt',
#        'db_cr_cd', 'payment_reporting_category', 'payment_category',
#        'is_international', 'default_brand', 'default_location', 'qrated_brand',
#        'coalesced_brand']]
y = list(df['Category_1'])

# X1, y = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)


# Random Forest
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
print( "Acc:", n/len(y_pred))


# xgboost
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
y_pred = xgbc.predict(X_test)
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
print( "Acc:", n/len(y_pred))




### Digging amt feature
decs = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
cates = ['Property and Business Services',
 'Travel',
 'Finance',
 'Services to Transport',
 'Education',
 'Communication Services',
 'Retail Trade',
 'Trade, Professional and Personal Services',
 'Health and Community Services',
 'Entertainment']
df_amt = pd.DataFrame([], columns= ['Category']+decs)
for c in cates:
    temp_d = dict(df[df['Category']==c]['amt'].describe())
    temp_d['Category'] = c
    df_temp = pd.DataFrame(temp_d,columns=['Category']+decs,index=[0])
    df_amt = pd.concat([df_amt, df_temp], axis=0)
df_amt.to_excel('Amount_digging.xlsx',index=False)


# # logistic
# 
# clf1 = LogisticRegression()
# clf1.fit(X_train, y_train)
# y_pred = clf1.predict(X_test)
# n=0
# for idx,yt in enumerate(y_test):
#     if yt==y_pred[idx]:
#         n+=1
# print( "Acc:", n/len(y_pred))


# SVM
# clf = SVC()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# n=0
# for idx,yt in enumerate(y_test):
#     if yt==y_pred[idx]:
#         n+=1
# print( "Acc:", n/len(y_pred))



# Check how max_depth affects overfit on RF
depths= list(np.arange(1,30,1))
acc_train = [0 for i in range(len(depths))]
acc_test = [0 for i in range(len(depths))]
for idx_d, d in enumerate(depths):
    clf1 = RandomForestClassifier(max_depth=d, random_state=0)
    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    y_pred_train = clf1.predict(X_train)
    n_train, n_test=0,0
    for idx,yt in enumerate(y_test):
        if yt==y_pred[idx]:
            n_test+=1
    acc_test[idx_d] = n_test/len(y_test)
    for idx,yt in enumerate(y_train):
        if yt ==y_pred_train[idx]:
            n_train+=1
    acc_train[idx_d] = n_train/len(y_train)
plt.plot(depths,acc_train,'s-',color = 'r',label="Train")#s-:方形
plt.plot(depths,acc_test,'o-',color = 'g',label="Test")#o-:圆形








### With language model information
df_similarity = pd.read_csv("similarity.csv",header=None)
s_cates=['Communication Services', 'Property and Business Services', 'Travel', 
         'Entertainment', 'Retail Trade', 'Services to Transport', 'Education', 
         'Health and Community Services', 'Trade, Professional and Personal Services', 
         'Finance']
rename_dict = dict(zip(list(df_similarity.columns), ["similarity_"+s for s in s_cates]))
df_similarity.rename(rename_dict, inplace=True, axis='columns')
# for col in df_similarity:
#     df_similarity[col] = df_similarity[col].apply(lambda x: pow(x,20))
# most_sim_cat = [-1 for i in range(len(df))]
# for idx in df_similarity.index:
#     temp = list(df_similarity.iloc[idx,:])
#     most_sim_cat[idx] =  hotdict[s_cates[temp.index(max(temp))]]
# df_raw['most_similar'] = most_sim_cat
# df_addl = pd.concat([df_raw, df_similarity], axis=1)

# Dealing with underflow
def simchar(x):
    if abs(x)<=0.05:
        return 0
    else:
        return x
for col in df_similarity:
    df_similarity[col] = df_similarity[col].apply(simchar)
df_addl = pd.concat([df_raw, df_similarity], axis=1)

# X1 = df_raw[['amt'] + cols + ['most_similar']]
X1 = df_addl[['amt'] + cols+list(df_similarity.columns)]

y = list(df_addl['Category_1'])


X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)


# Random Forest
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
print( "Acc:", n/len(y_pred))


# xgboost
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
y_pred = xgbc.predict(X_test)
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
print( "Acc:", n/len(y_pred))



