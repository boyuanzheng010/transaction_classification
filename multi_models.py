#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 20:22:48 2022

@author: Thomas_yanghan
"""
import pandas as pd
import numpy as np
import math
import pickle
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
import xgboost
from xgboost import XGBClassifier

df_raw = pd.read_csv('/Users/Thomas_yanghan/Desktop/wellsFargoAna/train_processed.csv')
df_similarity = pd.read_csv("similarities_big_lm.csv",header=None)

with open('./brand_embeddings_big.pkl', 'rb') as f:
    brandemb = pickle.load(f)
df_embeddings = pd.DataFrame(brandemb)


RFs={}
XGBs={}
errorIdx_RF={}
errorIdx_XGB={}
features_importance_RF={}
features_importance_XGB={}

ks = ['rawfeaturs_only', 'similarities_only', 'embeddings_only', 'rawfeatue_and_similarities']

y = list(df_raw['Category_1'])


# 0: without language model, raw features only
print("---*---\n Without language model, raw features only\n---*---")
cur = 0
df0 = df_raw.copy()
cols = ['sor_1', 'db_cr_cd_1', 'is_international_1', 'payment_category_1', 
        'state_1','merchant_cat_1']
X1 = df0[['amt'] + cols]
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)
testIdxes= list(X_test)

# Random Forest
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
importance_dict = dict(zip(X1.columns, clf.feature_importances_))
features_importance_RF[ks[cur]] = importance_dict
RFs[ks[cur]] = clf
y_pred = clf.predict(X_test)
errorIdxes = []
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
    else:
        errorIdxes.append(idx)
print( "Random Forest accuracy:", n/len(y_pred))
errorIdx_RF[ks[cur]] = errorIdxes


# xgboost
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
importance_dict = dict(zip(X1.columns, xgbc.feature_importances_))
features_importance_XGB[ks[cur]] = importance_dict
XGBs[ks[cur]] = xgbc
y_pred = xgbc.predict(X_test)
errorIdxes=[]
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
    else:
        errorIdxes.append(idx)
print( "Xgboost accuracy:", n/len(y_pred))
errorIdx_XGB[ks[cur]] = errorIdxes


# 1: Without raw features, similarities only
print("---*---\n Without raw features, similarities only \n---*---")
cur=1
s_cates=['Communication Services', 'Property and Business Services', 'Travel', 
         'Entertainment', 'Retail Trade', 'Services to Transport', 'Education', 
         'Health and Community Services', 'Trade, Professional and Personal Services', 
         'Finance']
rename_dict = dict(zip(list(df_similarity.columns), ["similarity_"+s for s in s_cates]))
df_similarity.rename(rename_dict, inplace=True, axis='columns')


# Dealing with underflow
def simchar(x):
    if abs(x)<=0.05:
        return 0
    else:
        return x
for col in df_similarity:
    df_similarity[col] = df_similarity[col].apply(simchar)
df1 = df_similarity.copy()
X1 = df1
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)
testIdxes= list(X_test)

# Random Forest
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
importance_dict = dict(zip(X1.columns, clf.feature_importances_))
features_importance_RF[ks[cur]] = importance_dict
RFs[ks[cur]] = clf 
y_pred = clf.predict(X_test)
errorIdxes = []
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
    else:
        errorIdxes.append(idx)
print( "Random Forest accuracy:", n/len(y_pred))
errorIdx_RF[ks[cur]] = errorIdxes


# xgboost
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
importance_dict = dict(zip(X1.columns, xgbc.feature_importances_))
features_importance_XGB[ks[cur]] = importance_dict
XGBs[ks[cur]] = xgbc
y_pred = xgbc.predict(X_test)
errorIdxes=[]
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
    else:
        errorIdxes.append(idx)
print( "Xgboost accuracy:", n/len(y_pred))
errorIdx_XGB[ks[cur]] = errorIdxes


# 2: Without raw features, embeddings only
print("---*---\n Without raw features, embeddings only\n---*---")
cur=2
rename_dict = dict(zip(list(df_embeddings.columns), ['embeddings_'+str(i) for i in list(df_embeddings.columns)]))

df2 = df_embeddings.rename(rename_dict, axis='columns')
X1 = df2
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)
testIdxes= list(X_test)

# Random Forest
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
importance_dict = dict(zip(X1.columns, clf.feature_importances_))
features_importance_RF[ks[cur]] = importance_dict
RFs[ks[cur]] = clf 
y_pred = clf.predict(X_test)
errorIdxes = []
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
    else:
        errorIdxes.append(idx)
print( "Random Forest accuracy:", n/len(y_pred))
errorIdx_RF[ks[cur]] = errorIdxes


# xgboost
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
importance_dict = dict(zip(X1.columns, xgbc.feature_importances_))
features_importance_XGB[ks[cur]] = importance_dict
XGBs[ks[cur]] = xgbc
y_pred = xgbc.predict(X_test)
errorIdxes=[]
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
    else:
        errorIdxes.append(idx)
print( "Xgboost accuracy:", n/len(y_pred))
errorIdx_XGB[ks[cur]] = errorIdxes



# 3: With raw features and similarities
print("---*---\n With raw features and similarities \n---*---")
cur=3
df_addl = pd.concat([df_raw, df_similarity], axis=1)

# X1 = df_raw[['amt'] + cols + ['most_similar']]
df3 = df_addl[['amt'] + cols + list(df_similarity.columns)]
X1=df3
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)
testIdxes= list(X_test)

# Random Forest
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
importance_dict = dict(zip(X1.columns, clf.feature_importances_))
features_importance_RF[ks[cur]] = importance_dict
RFs[ks[cur]] = clf 
y_pred = clf.predict(X_test)
errorIdxes = []
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
    else:
        errorIdxes.append(idx)
print( "Random Forest accuracy:", n/len(y_pred))
errorIdx_RF[ks[cur]] = errorIdxes


# xgboost
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
importance_dict = dict(zip(X1.columns, xgbc.feature_importances_))
features_importance_XGB[ks[cur]] = importance_dict
XGBs[ks[cur]] = xgbc
y_pred = xgbc.predict(X_test)
errorIdxes=[]
n=0
for idx,yt in enumerate(y_test):
    if yt==y_pred[idx]:
        n+=1
    else:
        errorIdxes.append(idx)
print( "Xgboost accuracy:", n/len(y_pred))
errorIdx_XGB[ks[cur]] = errorIdxes


results = {}
results["directory"] = ks
results["models"]={}
results["models"]["random_forest"] = RFs
results["models"]["xgboost"] = XGBs

results["error_indexes"]={}
results["error_indexes"]["random_forest"] = errorIdx_RF
results["error_indexes"]["xgboost"] = errorIdx_XGB

results["features_importance"]={}
results["error_indexes"]["random_forest"] = features_importance_RF
results["error_indexes"]["xgboost"] = features_importance_XGB


filename = 'multi_model_results.pkl'
pickle.dump(filename, open(filename, 'wb'))
















