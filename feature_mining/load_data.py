#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/1 15:33
# @Author  : sunday
# @Site    : 
# @File    : load_data.py
# @Software: PyCharm
class data(object):
    def load_data():
        train_ud_mdl = pd.read_csv('../data/lexin_train02/ud_mdl.csv', index_col=0)
        #未来六个月的消费信息及dep
        train_dep_mdl = pd.read_csv('../data/lexin_train02/dep_mdl.csv', index_col=0)
        #过去六个月用户场景行为信息
        train_login_scene_mdl = pd.read_csv('../data/lexin_train02/login_scene_mdl.csv', index_col=0)
        #过去六个月新增订单明细数据
        train_od_in6m_mdl = pd.read_csv('../data/lexin_train02/od_in6m_mdl.csv', index_col=0)
        #过去六个月订单行为汇总
        train_p6M_mdl = pd.read_csv('../data/lexin_train02/p6M_mdl.csv', index_col=0)
        #过去12个月月度订单金额
        train_p12M_mdl = pd.read_csv('../data/lexin_train02/p12M_mdl.csv', index_col=0)