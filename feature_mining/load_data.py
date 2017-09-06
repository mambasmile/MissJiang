#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/1 15:33
# @Author  : sunday
# @Site    : 
# @File    : load_data.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import csv
from sklearn.ensemble import BaggingRegressor

from  sklearn.model_selection import  cross_val_score

class data(object):
    def load_data(self):
        train_ud_mdl = pd.read_csv('../data/lexin_train02/ud_mdl.csv', index_col=0)
        #读取拼接后的数据
        path = unicode('../data/lexin_train02/训练数据集.csv', "utf8")
        intermediate_data = pd.read_csv(path, index_col=0)

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
        #数值类型进行log1p转换
        train_ud_mdl.loc["fstd_num"]=np.log1p(train_ud_mdl['fstd_num'])
        #进行one-hot编码
        all_ud_dummy_df = pd.get_dummies(train_ud_mdl,
                                         columns=['fsex', 'fis_entrance_exam', 'fcollege_level', 'fauth_source_type',
                                                  'fcal_graduation'])

        train_data = pd.concat((all_ud_dummy_df, intermediate_data), axis=1)

        # print train_data[5001]['dep']
        # print train_data.loc(5001)['dep']


        print train_data.isnull().sum()
        y_train = train_data['dep']
        mean_cols = train_data.mean()
        train_data = train_data.fillna(mean_cols)
        y_train = y_train.fillna(0).values


        train_data.pop('actual_od_brw_1stm')
        train_data.pop('actual_od_brw_2stm')
        train_data.pop('actual_od_brw_3stm')
        train_data.pop('actual_od_brw_4stm')
        train_data.pop('actual_od_brw_5stm')
        train_data.pop('actual_od_brw_6stm')
        train_data.pop('actual_od_brw_f6m')
        train_data.pop('dep')

        #删除非数值类型数据
        train_data.pop('fcredit_update_time201607')
        train_data.pop('fcredit_update_time201605')
        train_data.pop('fcredit_update_time201608')
        train_data.pop('fcredit_update_time201606')
        train_data.pop('fcredit_update_time201609')
        train_data.pop('fcredit_update_time201610')

        not_numeric_cols = train_data.columns[train_data.dtypes == 'object']
        print  not_numeric_cols
        print  train_data.columns
        X_train = train_data.values
        X_train = np.array(X_train)
        X_train = X_train.astype(float)
        # y_train = np.array(y_train)
        # y_train = y_train.astype(float)

        # 使用xgboost进行预测
        from xgboost import XGBClassifier
        params = [  3, 4, 5, 6,7,8,9,10]
        test_scores = []
        for param in params:
            clf = XGBClassifier(max_depth=param)
            # 进行交叉验证
            test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc'))
            test_scores.append(np.mean(test_score))
        import matplotlib.pyplot as plt
        plt.plot(params, test_scores)
        plt.title("max_depth vs roc_auc")
        plt.show()




        # print  res.columns
        # print res.head(10)

test = data()

test.load_data()