#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/1 15:33
# @Author  : sunday
# @Site    : 
# @File    : load_data.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import seaborn as sbn
import csv
from sklearn.ensemble import BaggingRegressor

from  sklearn.model_selection import  cross_val_score

class data(object):
    ud_mdl_file = unicode(r'../input/lexin_train02/特征处理后的ud_mdl.csv', 'utf-8')
    all_data_file = unicode(r'../input/lexin_train02/总的训练数据.csv', 'utf-8')


    def process_ud_mdl(self):
        """
        处理ud文件
        :return:
        """
        train_ud_mdl = pd.read_csv('../input/lexin_train02/ud_mdl.csv', index_col=0)



        train_ud_mdl["fauth_source_type"] = train_ud_mdl['fauth_source_type'].astype(str)
        train_ud_mdl["fsex"] = train_ud_mdl['fsex'].astype(str)
        # 删除没用的特征
        train_ud_mdl.pop('fregister_time')
        train_ud_mdl.pop('fpocket_auth_time')
        train_ud_mdl.pop('fdomicile_provice')
        train_ud_mdl.pop('fdomicile_city')
        train_ud_mdl.pop('fdomicile_area')
        train_ud_mdl.pop('sch_fprovince_name')
        train_ud_mdl.pop('sch_fcity_name')
        train_ud_mdl.pop('sch_fregion_name')
        train_ud_mdl.pop('sch_fcompany_name')
        train_ud_mdl.pop('fschoolarea_name_md5')
        # 数值类型进行log1p转换
        train_ud_mdl["fstd_num"] = np.log1p(train_ud_mdl['fstd_num'])
        # 进行one-hot编码
        all_ud_dummy_df = pd.get_dummies(train_ud_mdl,
                                         columns=['fsex', 'fis_entrance_exam', 'fcollege_level', 'fauth_source_type',
                                                  'fcal_graduation'])
        mean_cols = all_ud_dummy_df.mean()
        all_ud_dummy_df = all_ud_dummy_df.fillna(mean_cols)
        print all_ud_dummy_df.isnull().sum().sort_values(ascending=False)

        #存入文件
        all_ud_dummy_df.to_csv(self.ud_mdl_file, index=True)



    def process_intermediate_data(self):
        """
        处理曹进的数据和ud数据并保存文件
        :return:
        """
        # 读取拼接后的数据
        path = unicode('../input/lexin_train02/训练数据集.csv', "utf8")
        intermediate_data = pd.read_csv(path, index_col=0)
        print intermediate_data.columns


        not_numeric_cols = intermediate_data.columns[intermediate_data.dtypes == 'object']
        for x in not_numeric_cols:
            intermediate_data.pop(x)

        # print intermediate_data.columns
        # print intermediate_data.shape

        print intermediate_data.isnull().sum().sort_values(ascending=False)
        ud_mdl_data = pd.read_csv(self.ud_mdl_file,index_col=0)
        print ud_mdl_data.shape
        all_data = pd.concat((ud_mdl_data,intermediate_data),axis=1)
        print all_data.shape
        print all_data.isnull().sum().sort_values(ascending=False)
        all_data.to_csv(self.all_data_file, index=True)


    def test_xgboost(self):

        all_data = pd.read_csv(self.all_data_file,index_col=0)
        all_data.pop('actual_od_brw_1stm')
        all_data.pop('actual_od_brw_2stm')
        all_data.pop('actual_od_brw_3stm')
        all_data.pop('actual_od_brw_4stm')
        all_data.pop('actual_od_brw_5stm')
        all_data.pop('actual_od_brw_6stm')
        all_data.pop('actual_od_brw_f6m')
        y_train = all_data.pop("dep")


        X_train = all_data.values


        # not_numeric_cols = train_data.columns[train_data.dtypes == 'object']
        # print  not_numeric_cols
        # print  train_data.columns
        # X_train = train_data.values
        # X_train = np.array(X_train)
        # X_train = X_train.astype(float)


        # 使用xgboost进行预测
        from xgboost import XGBRegressor
        params = [ 3, 4, 5,6,7,8,9,10]
        test_scores = []
        for param in params:
            clf = XGBRegressor(max_depth=param)
            # 进行交叉验证
            test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc'))
            test_scores.append(np.mean(test_score))

        import matplotlib.pyplot as plt
        plt.plot(params, test_scores)
        plt.title("max_depth vs roc_auc")
        plt.show()



data_process = data()
data_process.process_ud_mdl()














#   使用GBM进行预测

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