#coding=utf-8

import pandas as pd

# p6m_DF1 = pd.read_csv('../train_file/p6m挖掘出的新特征.csv')
# p6m_DF2 = pd.read_csv('../test_file/p6m挖掘出的新特征.csv')
p6m_DF1 = pd.read_csv('../train_file/特征处理后的p6M_mdl.csv')
p6m_DF2 = pd.read_csv('../test_file/经过特征处理后的p6M_offtime1.csv')
p6m_DF = pd.concat([p6m_DF1,p6m_DF2],axis=0)
features = pd.read_csv('../train_file/p6M_mdl.csv').columns





featureLs = p6m_DF.columns


for val in features:
    if val in ['fuid_md5','pyear_month','cyc_date','fcredit_update_time']:
        continue
    sum = p6m_DF[val + 'JUN16']
    for fea in featureLs:
        if fea[:-5] == val and fea != (val+'JUN16'):
            sum+=p6m_DF[fea]
    p6m_DF['avg'+val] = sum/6.0

p6m_DF = p6m_DF.fillna(0)

p6m_DF.iloc[:50000,:].to_csv('../train_file/求平均值后p6M_mdl(1).csv',index=False)
p6m_DF.iloc[50000:,:].to_csv('../test_file/求平均值后p6M_offtime(1).csv',index=False)


