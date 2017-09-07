#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-7 下午4:59
# @Author  : sunday
# @Site    : 
# @File    : GBDT.py
# @Software: PyCharm


# Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics  # Additional     scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pylab as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


class new_test():
    all_data_file = unicode(r'../input/lexin_train02/总的训练数据.csv', 'utf-8')
    train = pd.read_csv(all_data_file, index_col=0)
    train.pop('actual_od_brw_1stm')
    train.pop('actual_od_brw_2stm')
    train.pop('actual_od_brw_3stm')
    train.pop('actual_od_brw_4stm')
    train.pop('actual_od_brw_5stm')
    train.pop('actual_od_brw_6stm')
    train.pop('actual_od_brw_f6m')

    target = 'dep'
    IDcol = 'fuid_md5'

    def modelfit(self, alg, dtrain, dtest, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
        # Fit the algorithm on the data
        alg.fit(dtrain[predictors], dtrain['dep'])

        # Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors])
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

        # Perform cross-validation:
        if performCV:
            cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['dep'], cv=cv_folds,
                                                        scoring='roc_auc')

        # Print model report:
        print "\nModel Report"
        print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['dep'].values, dtrain_predictions)
        print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['dep'], dtrain_predprob)

        if performCV:
            print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
            np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

        # Print Feature Importance:
        if printFeatureImportance:
            feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.show()






    def run(self):


        # # Choose all predictors except target & IDcols
        # predictors = [x for x in self.train.columns if x not in [self.target, self.IDcol]]
        # gbm0 = GradientBoostingClassifier(random_state=10)
        # self.modelfit(gbm0, self.train, test, predictors)




        # #Step 1- Find the number of estimators for a high learning rate
        # # Choose all predictors except target & IDcols
        # predictors = [x for x in self.train.columns if x not in [self.target, self.IDcol]]
        # param_test1 = {'n_estimators': range(20, 81, 10)}
        # gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,
        #                                                              min_samples_leaf=50, max_depth=8,
        #                                                              max_features='sqrt', subsample=0.8,
        #                                                              random_state=10),
        #                         param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
        # gsearch1.fit(self.train[predictors],self.train[self.target])
        # print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_



        #Step 2- Tune tree-specific parameters
        # Grid seach on subsample and max_features
        predictors = [x for x in self.train.columns if x not in [self.target, self.IDcol]]
        # param_test2 = {'max_depth': range(5, 16, 2), 'min_samples_split': range(200, 1001, 200)}
        # gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,
        #                                                              max_features='sqrt', subsample=0.8,
        #                                                              random_state=10),
        #                         param_grid=param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
        # gsearch2.fit(self.train[predictors], self.train[self.target])
        # print gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


        #grid search on subsample and max features
        # param_test3 = {'min_samples_split': range(1000, 2100, 200), 'min_samples_leaf': range(30, 71, 10)}
        # gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_depth=13,
        #                                                              max_features='sqrt', subsample=0.8,
        #                                                              random_state=10),
        #                         param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
        # gsearch3.fit(self.train[predictors],self.train[self.target])
        # print gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


        # Grid seach on subsample and max_features
        # param_test4 = {'max_features': range(7, 100, 2)}
        # gsearch4 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_depth=13,
        #                                                              min_samples_split=1400, min_samples_leaf=70,
        #                                                              subsample=0.8, random_state=10),
        #                         param_grid=param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
        # gsearch4.fit(self.train[predictors], self.train[self.target])
        # print gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_



        # Grid seach on subsample and max_features
        # param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9,0.95]}
        # gsearch5 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_depth=13,
        #                                                              min_samples_split=1400, min_samples_leaf=70,
        #                                                              subsample=0.8, random_state=10, max_features=17),
        #                         param_grid=param_test5, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
        # gsearch5.fit(self.train[predictors], self.train[self.target])
        # print gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
        # self.modelfit(gsearch5.best_estimator_, self.train, test, predictors)

        gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_depth=13,
                                   min_samples_split=1400, min_samples_leaf=70,
                                   subsample=0.9, random_state=10, max_features=17)
        self.modelfit(gbm_tuned_1,self.train ,test, predictors)


test = new_test()
# test.run()