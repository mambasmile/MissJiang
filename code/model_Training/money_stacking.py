#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午8:06
# @Author  : sunday
# @Site    : 
# @File    : money_stacking.py
# @Software: PyCharm
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score,mean_absolute_error
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd
from xgboost import XGBClassifier
from sklearn import preprocessing
from create_submission import create_submission
future_6_month = unicode(r'../output/未来6个月逾期金额.csv', 'utf-8')
all_data_file = unicode(r'../input/lexin_train02/linear训练数据集特征挖掘1.0.csv', 'utf-8')
train_X = pd.read_csv(all_data_file)
dep_mdl_path = unicode(r'../input/lexin_train02/dep_mdl.csv','utf-8')
train_y = pd.read_csv(dep_mdl_path)
test_data_file = unicode(r'../input/lexin_test/验证数据集_4/linear测试数据集特征挖掘1.0.csv', 'utf-8')
test_data = pd.read_csv(test_data_file)
train_data = pd.merge(train_X,train_y,on="fuid_md5")
train_data.sample(frac=1)
train_data.pop('actual_od_brw_1stm')
train_data.pop('actual_od_brw_2stm')
train_data.pop('actual_od_brw_3stm')
train_data.pop('actual_od_brw_4stm')
train_data.pop('actual_od_brw_5stm')
train_data.pop('actual_od_brw_6stm')
train_data.pop('dep')
target = 'actual_od_brw_f6m'
IDcol = 'fuid_md5'

predictors = [x for x in train_data.columns if x not in [target, IDcol]]


'''创建训练的数据集'''
data = train_data[predictors].values
target = train_data['actual_od_brw_f6m'].values
# data, target = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=0.60)

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

'''模型融合中使用到的各个单模型'''
clfs = [
        RandomForestRegressor(n_estimators=120, max_features=26),
        XGBRegressor(
            learning_rate=0.1,
            n_estimators=140,
            max_depth=7,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:linear',
            nthread=4,
            scale_pos_weight=1,
            seed=27
        )
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

X, X_predict, y, y_predict = train_test_split(preprocessing.normalize(all_data[0:50000][predictors].values), train_data['actual_od_brw_f6m'].values, test_size=0.33, random_state=2017)



# X, y = preprocessing.normalize(all_data[0:50000][features].values),train_data['dep'].values
# X_predict = preprocessing.normalize(all_data[50000:100000][features].values)


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
        y_submission = clf.predict(X_test)
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict(X_predict)
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print("val mae Score: %f" % mean_absolute_error(y_predict, dataset_blend_test[:, j]))
# clf = LogisticRegression()
# clf2 = XGBRegressor(learning_rate=0.05, subsample=0.5, max_depth=4, n_estimators=90)
# clf = GradientBoostingRegressor(learning_rate=0.02, subsample=0.5, max_depth=5, n_estimators=40)
clf =Ridge()
clf.fit(dataset_blend_train, y)
y_submission = clf.predict(dataset_blend_test)

print("Linear stretch of predictions to [0,1]")
print("blend result")
submission = pd.DataFrame(data={"Id": test_data.fuid_md5, "prob": y_submission})
submission.to_csv(future_6_month, index=False)
# create_submission("5.0-特征权重4-删除特征2.0")
print("val mae Score: %f" % (mean_absolute_error(y_predict, y_submission)))
