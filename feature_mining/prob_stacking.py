#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-18 下午9:46
# @Author  : sunday
# @Site    : 
# @File    : prob_stacking.py
# @Software: PyCharm
# coding=utf8


from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd
from xgboost import XGBClassifier
from sklearn import preprocessing
from create_submission import create_submission
prob_path = unicode(r'../output/逾期概率.csv', 'utf-8')
all_data_file = unicode(r'../input/lexin_train02/训练数据5.0.csv', 'utf-8')
train_X = pd.read_csv(all_data_file)
dep_mdl_path = unicode(r'../input/lexin_train02/dep_mdl.csv','utf-8')
train_y = pd.read_csv(dep_mdl_path)
test_data_file = unicode(r'../input/lexin_test/验证数据集_4/测试数据5.0.csv', 'utf-8')
test_data = pd.read_csv(test_data_file)
train_data = pd.merge(train_X,train_y,on="fuid_md5")
train_data.sample(frac=1)
train_data.pop('actual_od_brw_1stm')
train_data.pop('actual_od_brw_2stm')
train_data.pop('actual_od_brw_3stm')
train_data.pop('actual_od_brw_4stm')
train_data.pop('actual_od_brw_5stm')
train_data.pop('actual_od_brw_6stm')
train_data.pop('actual_od_brw_f6m')
target = 'dep'
IDcol = 'fuid_md5'

predictors = [x for x in train_data.columns if x not in [target, IDcol]]
feature_imp_path = unicode(r'../input/特征权重4.csv', 'utf-8')
features = []
with open(feature_imp_path, 'r') as fea_imp:
    for i in fea_imp:
        lists = i.split(',')
        features.append(lists[0])

'''创建训练的数据集'''
data = train_data[features].values
target = train_data['dep'].values
# data, target = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=0.60)

'''模型融合中使用到的各个单模型'''
clfs = [RandomForestClassifier(n_estimators=140,max_depth=11,min_samples_leaf=15, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, n_estimators=60, max_depth=9,
                                   min_samples_split=1000, min_samples_leaf=70,
                                   subsample=0.9, random_state=10, max_features=20),
        XGBClassifier(learning_rate=0.111, n_estimators=140, max_depth=3, min_child_weight=6,
                      gamma=0, subsample=0.6, colsample_bytree=0.9, objective='binary:logistic',
                      nthread=4, scale_pos_weight=1, seed=27)
        ]

'''切分一部分数据作为测试集'''
all_data = pd.concat((train_data,test_data),axis=0)

from sklearn.preprocessing import Imputer

"""正则化处理等"""
# all_data = Imputer().fit_transform(all_data[predictors])
# all_data = pd.DataFrame(all_data)
# all_data = all_data.fillna(0)
# normalize the data attributes
# normalized_X = preprocessing.normalize(all_data)
# # standardize the data attributes
# standardized_X = preprocessing.scale(X)

# X, X_predict, y, y_predict = train_test_split(preprocessing.normalize(all_data[0:50000][predictors].values), train_data['dep'].values, test_size=0.33, random_state=2017)



X, y = preprocessing.normalize(all_data[0:50000][features].values),train_data['dep'].values
X_predict = preprocessing.normalize(all_data[50000:100000][features].values)


dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

'''5折stacking'''
n_folds = 5
skf = list(StratifiedKFold(y, n_folds,shuffle=True))
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        # print("Fold", i)
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    # print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))
# clf = LogisticRegression()
clf2 = XGBClassifier(learning_rate=0.05, subsample=0.5, max_depth=4, n_estimators=90)
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=5, n_estimators=40)
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

print("Linear stretch of predictions to [0,1]")
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
print("blend result")
submission = pd.DataFrame(data={"Id": test_data.fuid_md5, "prob": y_submission})
submission.to_csv(prob_path, index=False)
create_submission("5.0-特征权重4-删除特征2.0")
# print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))