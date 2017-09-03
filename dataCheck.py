#coding=utf-8

import pandas as pd
import numpy as np
from utils import utils
import matplotlib.pyplot as plt

login_scene_mdl = unicode(r'D:\牛客网比赛\lexin_train02\login_scene_mdl.csv','utf-8')
all_login_scene_mdl_file = unicode(r'D:\牛客网比赛\lexin_train02\特征处理后的login_scene_md_l.csv','utf-8')
login_scene_mdl_file = unicode(r'D:\牛客网比赛\lexin_train02\特征处理后的login_scene_mdl.csv','utf-8')
login_scene_mdl_info = unicode(r'D:\牛客网比赛\lexin_train02\login_scene_mdl详细数据.csv','utf-8')

od_in6m_mdl = unicode(r'D:\牛客网比赛\lexin_train02\od_in6m_mdl.csv','utf-8')
od_in6m_mdl_info = unicode(r'D:\牛客网比赛\lexin_train02\od_in6m_mdl详细信息.csv','utf-8')

p6M_mdl = unicode(r'D:\牛客网比赛\lexin_train02\p6M_mdl.csv','utf-8')
all_p6M_mdl_file = unicode(r'D:\牛客网比赛\lexin_train02\特征处理后的p6M_mdl_11.csv','utf-8')

p6M_mdl_file = unicode(r'D:\牛客网比赛\lexin_train02\特征处理后的p6M_mdl.csv','utf-8')
p6M_mdl_info = unicode(r'D:\牛客网比赛\lexin_train02\p6M_mdl详细信息.csv','utf-8')

p12M_mdl = unicode(r'D:\牛客网比赛\lexin_train02\p12M_mdl.csv','utf-8')
all_p12M_mdl_file = unicode(r'D:\牛客网比赛\lexin_train02\特征处理后的p12M_mdl_1.csv','utf-8')
p12M_mdl_info = unicode(r'D:\牛客网比赛\lexin_train02\p12M_mdl详细信息.csv','utf-8')

dep_mdl = unicode(r'D:\牛客网比赛\lexin_train02\dep_mdl.csv','utf-8')
dep_mdl_info = unicode(r'D:\牛客网比赛\lexin_train02\dep_mdl详细信息.csv','utf-8')

ud_mdl = unicode(r'D:\牛客网比赛\lexin_train02\ud_mdl.csv','utf-8')
ud_mdl_info = unicode(r'D:\牛客网比赛\lexin_train02\ud_mdl详细信息.csv','utf-8')

# data_login_scene = pd.read_csv(login_scene_mdl)
# print data_login_scene.columns
# print data_login_scene.info()
# data_login_scene.describe().to_csv(ud_mdl_info)
# print data_login_scene['dep'].value_counts()
# print data_login_scene['c_scene_sh_min_dur'].value_counts()

# order_dataFrame = pd.read_csv(od_in6m_mdl)


"""处理过去12个月的数据"""
# data_p12M_mdl = pd.read_csv(p12M_mdl)
# cvc_data = data_p12M_mdl.iloc[:12,:]['cyc_date']
# all_data_p12M = utils.rowTransform(data_p12M_mdl.iloc[:12,:],cvc_data,0)
# for i in xrange(1,50000):
#     tmpDataFrame = utils.rowTransform(data_p12M_mdl.iloc[i*12:(i+1)*12,:],cvc_data,i)
#     try:
#         all_data_p12M = pd.concat((all_data_p12M,tmpDataFrame),axis=0)
#     except AssertionError,e:
#         print i
#
# all_data_p12M.to_csv(all_p12M_mdl,index=False)

"""处理过去6个月的数据"""
# all_data_p12M = pd.read_csv(all_p12M_mdl_file)
#
# data_p6M_mdl = pd.read_csv(p6M_mdl)
# rowName = all_data_p12M['fuid_md5']
#
# all_data_p6M_mdl = utils.processP6M(data_p6M_mdl,rowName)
# all_data_p6M_mdl.to_csv(all_p6M_mdl_file,index=False)
#
# """处理login_scene数据"""
# all_data_p12M = pd.read_csv(all_p12M_mdl_file)
# rowName = all_data_p12M['fuid_md5']
# data_login_scene_mdl = pd.read_csv(login_scene_mdl)
# all_login_scene_mdl = utils.processP6M(data_login_scene_mdl,rowName)
# all_login_scene_mdl.to_csv(all_login_scene_mdl_file,index=False)

"""特征挖掘----过去12个月的订单金额"""
# file = unicode(r'D:\牛客网比赛\lexin_train02\过去12个月月度订单金额与用户逾期还款信息.csv','utf-8')
# all_p12M_mdl = pd.read_csv(all_p12M_mdl_file)
# data_dep_mdl = pd.read_csv(dep_mdl)
# res_DataFrame = pd.merge(all_p12M_mdl,data_dep_mdl,on='fuid_md5')
# res_DataFrame.to_csv(file,index=False)

# res_DataFrame = pd.read_csv(file)
# col_str = 'od_brw201511,od_brw201512,od_brw201601,od_brw201602,od_brw201603,od_brw201604,od_brw201605,od_brw201606,od_brw201607,od_brw201608,od_brw201609,od_brw201610'
# col_index = col_str.split(',')
# for col in col_index:
#     res_DataFrame[col] = np.log1p(res_DataFrame[col])
# res_DataFrame.to_csv(file,index=False)

"""特征挖掘----过去六个月用户场景行为信息"""
# p6M_resDataFrame = pd.read_csv(p6M_mdl_file)
# # p6M_resDataFrame.to_csv(p6M_mdl_file,index=False)
# columns = p6M_resDataFrame.columns
# p6M_resDataFrame.pop(columns[0]).to_csv(p6M_mdl_file,index=False)

"""组合过去12个月月度订单金额和用户逾期还款信息和用户登录信息"""
# outFile = unicode(r'D:\牛客网比赛\lexin_train02\过去12个月月度订单金额和用户逾期还款信息和用户登录信息的汇总.csv','utf-8')
# login_resDataFrame = pd.read_csv(login_scene_mdl_file)
# all_p12MandDep_file = pd.read_csv(file)
# res_DataFrame = pd.merge(all_p12MandDep_file,login_resDataFrame,on='fuid_md5',suffixes=('_login','_order'))
# res_DataFrame.to_csv(outFile,index=False)

"""组合过去12个月月度订单金额和用户逾期还款信息和用户登录信息和过去6个月订单行为信息"""
resultFile = unicode(r'D:\牛客网比赛\lexin_train02\过去12个月月度订单金额和用户逾期还款信息和用户登录信息和过去6个月订单行为信息的汇总.csv','utf-8')
# all_p12_login_dep_DataFrame = pd.read_csv(outFile)
# all_p6_DataFrame = pd.read_csv(all_p6M_mdl_file)
# all_p6_DataFrame = all_p6_DataFrame.drop(['od_brw201607','od_brw201605','od_brw201606','od_brw201608','od_brw201609','od_brw201610'],axis=1)
# res_DataFrame = pd.merge(all_p12_login_dep_DataFrame,all_p6_DataFrame,on='fuid_md5',suffixes=('','p6'))
# res_DataFrame.to_csv(resultFile,index=False)

"""特征挖掘"""
# order12_col = pd.read_csv(p6M_mdl).columns
# order6_col = pd.read_csv(p12M_mdl).columns
# userDep_col = pd.read_csv(dep_mdl).columns
# userLogin_col = pd.read_csv(login_scene_mdl).columns
#
# featureLs = order12_col.values.tolist()
# featureLs.extend(order6_col.values.tolist())
# featureLs.extend(userDep_col.values.tolist())
# featureLs.extend(userLogin_col.values.tolist())
#
# res_feature = []
# for val in featureLs:
#     if val == 'pyear_month' or val == 'cyc_date':
#         continue
#     elif val == 'fuid_md5':
#         res_feature.append(val)
#     else:
#         val+='.*'
#         res_feature.append(val)
# regix = "|".join(res_feature)
# # print regix

file = unicode(r'D:\牛客网比赛\训练数据集.csv','utf-8')
# allDataFrame = pd.read_csv(resultFile)
# allDataFrame.filter(regex=regix).to_csv(file,index=False)

allDataFrame = pd.read_csv(file)
columnsLs = ['fopen_to_buy201607','fopen_to_buy201605','fopen_to_buy201608','fopen_to_buy201606',
             'fopen_to_buy201609','fopen_to_buy201610']
print allDataFrame['fopen_to_buy201607'].head(5)
for val in columnsLs:
        allDataFrame[val] = np.log1p(allDataFrame[val])
# allDataFrame.drop(['fcredit_update_time201607','fcredit_update_time201605',
#                    'fcredit_update_time201608','fcredit_update_time201606',
#                    'fcredit_update_time201609','fcredit_update_time201610'],axis=1).to_csv(file)
print allDataFrame['fopen_to_buy201607'].head(5)