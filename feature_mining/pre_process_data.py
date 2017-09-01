#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/31 20:55
# @Author  : sunday
# @Site    : 
# @File    : pre_process_data.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

def pre_process_ud(train_ud_mdl):
    train_ud_mdl = pd.read_csv('../data/lexin_train02/ud_mdl.csv',index_col=0)

    # #未来六个月的消费信息及dep
    # train_dep_mdl = pd.read_csv('../data/lexin_train02/dep_mdl.csv', index_col=0)
    # #过去六个月用户场景行为信息
    # train_login_scene_mdl = pd.read_csv('../data/lexin_train02/login_scene_mdl.csv', index_col=0)
    # #过去六个月新增订单明细数据
    # train_od_in6m_mdl = pd.read_csv('../data/lexin_train02/od_in6m_mdl.csv', index_col=0)
    #过去六个月订单行为汇总
    train_p6M_mdl = pd.read_csv('../data/lexin_train02/p6M_mdl.csv', index_col=0)
    # #过去12个月月度订单金额
    # train_p12M_mdl = pd.read_csv('../data/lexin_train02/p12M_mdl.csv', index_col=0)

    train_ud_mdl["fauth_source_type"] = train_ud_mdl['fauth_source_type'].astype(str)
    train_ud_mdl["fsex"] = train_ud_mdl['fsex'].astype(str)
    #删除没用的特征
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

    plt.show(np.log1p(train_ud_mdl['fstd_num']).hist())
    train_ud_mdl.loc["fstd_num"]=np.log1p(train_ud_mdl['fstd_num'])
    plt.show(np.log1p(train_ud_mdl['fstd_num']).hist())
    # print  pd.get_dummies(train_ud_mdl["fauth_source_type"],prefix="fauth_source_type").head()

    #进行one-hot编码
    all_ud_dummy_df = pd.get_dummies(train_ud_mdl,columns=['fsex','fis_entrance_exam','fcollege_level','fauth_source_type','fcal_graduation'])











    print all_ud_dummy_df.columns
    #检查数据缺失情况
    print all_ud_dummy_df.isnull().sum().sort_values(ascending=False).head(15)
    print  all_ud_dummy_df.isnull().sum().sum()
    numeric_cols = all_ud_dummy_df.columns[all_ud_dummy_df.dtypes != 'object']
    print numeric_cols
    #按照第一列纵向合并数据
    # res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
    # print  train_ud_mdl.columns

pre_process()