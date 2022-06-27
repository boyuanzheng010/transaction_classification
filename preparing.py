#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:57:37 2022

@author: Thomas_yanghan
"""
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


df = pd.read_excel("CAC_2022_Training+Data+Set+New.xlsx")


COLS = ['sor', 'cdf_seq_no', 'trans_desc', 'merchant_cat_code', 'amt',
       'db_cr_cd', 'payment_reporting_category', 'payment_category',
       'is_international', 'default_brand', 'default_location', 'qrated_brand',
       'coalesced_brand', 'Category']

## Preprocessing
### Dealing with default location, to state
dloc = df['default_location'].apply(lambda x: x.split(" ")[-1] if pd.notnull(x) else "NA")
dloc1 = dloc.apply(lambda x: x if len(x)==2 else "NA")
df['state'] = dloc1

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

# Random Forest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
X1 = df[['amt'] + cols]
# X2 = df[['sor', 'cdf_seq_no', 'trans_desc', 'merchant_cat_code', 'amt',
#        'db_cr_cd', 'payment_reporting_category', 'payment_category',
#        'is_international', 'default_brand', 'default_location', 'qrated_brand',
#        'coalesced_brand']]
y = list(df['Category_1'])

# X1, y = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(max_depth=18, random_state=0)
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
# from sklearn.linear_model import LogisticRegression
# clf1 = LogisticRegression()
# clf1.fit(X_train, y_train)
# y_pred = clf1.predict(X_test)
# n=0
# for idx,yt in enumerate(y_test):
#     if yt==y_pred[idx]:
#         n+=1
# print( "Acc:", n/len(y_pred))











