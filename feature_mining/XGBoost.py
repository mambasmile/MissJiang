#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-7 下午5:00
# @Author  : sunday
# @Site    : 
# @File    : XGBoost.py
# @Software: PyCharm

#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor

from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
test_data_file = unicode(r'../input/lexin_test/验证数据集_4/总的测试数据.csv', 'utf-8')
test_data = pd.read_csv(test_data_file,index_col=0)
prob_path = unicode(r'../input/lexin_test/验证数据集_4/逾期概率.csv', 'utf-8')
future_6_month = unicode(r'../input/lexin_test/验证数据集_4/未来6个月逾期金额.csv', 'utf-8')
all_submit_path =  unicode(r'../input/lexin_test/验证数据集_4/提交文件.txt', 'utf-8')
all_data_file = unicode(r'../input/lexin_train02/总的训练数据.csv', 'utf-8')
X_train = pd.read_csv(all_data_file, index_col=0)

dep_mdl_path = unicode(r'../input/lexin_train02/dep_mdl.csv','utf-8')
y_train = pd.read_csv(dep_mdl_path)

train_data = pd.merge(X_train,y_train,on="fuid_md5")
train_data.pop('actual_od_brw_1stm')
train_data.pop('actual_od_brw_2stm')
train_data.pop('actual_od_brw_3stm')
train_data.pop('actual_od_brw_4stm')
train_data.pop('actual_od_brw_5stm')
train_data.pop('actual_od_brw_6stm')
train_data.pop('actual_od_brw_f6m')

target = 'dep'
IDcol = 'fuid_md5'


class xgboost():

    def modelfit(self,alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
        if useTrainCV:
            xgb_param = alg.get_xgb_params()
            xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                              metrics='mae', early_stopping_rounds=early_stopping_rounds)
            alg.set_params(n_estimators=cvresult.shape[0])

        # Fit the algorithm on the data
        alg.fit(dtrain[predictors], dtrain[target], eval_metric='mae')

        """
        predict future dep prob and print the auc and accuracy 
        """
        # Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors])
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
        # predict test set:
        dtest_predprob = alg.predict_proba(test_data[predictors])[:, 1]
        #store to file
        submission = pd.DataFrame(data={"Id": test_data.fuid_md5, "prob": dtest_predprob})
        submission.to_csv(future_6_month, index=False)
        # Print model report:
        print "\nModel Report"
        print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
        print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)


        """
        predict the future 6 month spend money
        """
        # # Predict training set:
        # dtrain_predictions = alg.predict(dtrain[predictors])
        # #predict test set:
        # dtest_predictions = alg.predict(test_data[predictors])
        # #store to file
        # submission = pd.DataFrame(data={"Id": test_data.fuid_md5, "prob": dtest_predictions})
        # submission.to_csv(future_6_month,index=False)
        # # Print model report:
        # print "\nModel Report"
        # print "MAE (Train): %f" % metrics.mean_absolute_error(dtrain[target], dtrain_predictions)



        feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()




    def run(self):
        # Choose all predictors except target & IDcols
        predictors = [x for x in train_data.columns if x not in [target, IDcol]]
        xgb1 = XGBClassifier(
            learning_rate=0.1,
            n_estimators=1000,
            max_depth=6,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=27)
        xgb2 = XGBRegressor(

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
        # self.modelfit(xgb2, train_data, predictors)


        """
        choose the premeter for xgbregressor
        """
        # param_test1 = {
        #  'max_depth':range(5,10,2),
        #  'min_child_weight':range(1,6,2)
        # }
        # gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate =0.1, n_estimators=140, max_depth=7,
        # min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
        #  objective= 'reg:linear', nthread=4,     scale_pos_weight=1, seed=27),
        #  param_grid = param_test1,     scoring='mean_absolute_error',n_jobs=4,iid=False, cv=5)
        # gsearch1.fit(train_data[predictors],train_data[target])
        # print gsearch1.grid_scores_, gsearch1.best_params_,     gsearch1.best_score_


        """
        choose the premeter for xgbclassifier
        """
        param_test1 = {
         # 'n_estimators':range(100,200,1000),
         # 'min_child_weight':range(1,6,2)
        }
        gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5,
        min_child_weight=5, gamma=0, subsample=0.8,             colsample_bytree=0.8,
         objective= 'binary:logistic', nthread=4,     scale_pos_weight=1, seed=27),
         param_grid = param_test1,     scoring='roc_auc',n_jobs=4,iid=False, cv=5)
        gsearch1.fit(train_data[predictors],train_data[target])
        print gsearch1.grid_scores_, gsearch1.best_params_,     gsearch1.best_score_



    def submission(self):
        """
        merge two submission part
        :return:
        """
        prob = pd.read_csv(prob_path)
        money = pd.read_csv(future_6_month)
        all_submit_data = pd.merge(prob,money,on="Id")
        all_submit_data.to_csv(all_submit_path,sep=' ',columns=None,index_label="", index=False)

test = xgboost()
test.run()