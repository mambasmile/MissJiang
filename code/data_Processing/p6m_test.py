#coding=utf-8

import pandas as pd
import numpy as np

p6m_DF1 = pd.read_csv('../train_file/特征处理后的p6M_mdl.csv')
p6m_DF2 = pd.read_csv('../test_file/经过特征处理后的p6M_offtime1.csv')

p6m_DF = pd.concat([p6m_DF1,p6m_DF2],axis=0)
# print p6m_DF1.shape
# print p6m_DF2.shape
print p6m_DF.shape

result = p6m_DF[['fuid_md5']]
featureLs = ['od_cnt','actual_od_cnt','virtual_od_cnt','od_zdfq_cnt','od_brw',
             'actual_od_brw','virtual_od_brw','od_zdfq_brw','cumu_od_cnt','cumu_actual_od_cnt'
            ,'cumu_virtual_od_cnt','cumu_od_zdfq_cnt','cumu_od_brw','cumu_actual_od_brw',
             'cumu_virtual_od_brw','cumu_od_zdfq_brw','payed_capital','payed_actual_capital',
             'payed_virtual_capital','payed_zdfq_capital','payed_mon_fee','payed_zdfq_mon_fee',
             'payed_tot_fee','payed_zdfq_tot_fee','bal','zdfq_bal','paying_mon_fee','zdfq_paying_mon_fee'
             ,'paying_tot_fee','zdfq_paying_tot_fee','paying_complete_od_cnt','payed_complete_od_cnt',
             'paying_complete_od_brw','payed_complete_od_brw',
             'acre_repay_od_cnt','acre_repay_od_cpt','foverdue_paying_day',
             'foverdue_paying_cyc','foverdue_payed_day','foverdue_payed_cyc','cpt_pymt','credit_limit',
             'fcredit_update_time','futilization','fopen_to_buy']
for mon in ['JUL16','OCT16','AUG16','SEP16','JUN16','MAY16']:
    for fea in featureLs:
        result[fea+mon] = p6m_DF[fea+mon]

for mon in ['JUL16','OCT16','AUG16','SEP16','JUN16','MAY16']:
    ##应还总服务费-已还总服务费
    result['dh_tot_fee' + mon] = p6m_DF['paying_tot_fee' + mon] - p6m_DF['payed_tot_fee' + mon]
    ##待还总服务费所占比率
    result['dh_tot_feePercent' + mon] = result['dh_tot_fee' + mon].divide(p6m_DF['paying_tot_fee' + mon],fill_value = 0)
    ##应还月服务费-已还月服务费
    result['dh_mon_fee' + mon] = p6m_DF['paying_mon_fee' + mon] - p6m_DF['payed_mon_fee' + mon]
    ##待还月服务费所占比率
    result['dh_mon_feePercent' + mon] = result['dh_mon_fee' + mon].divide(p6m_DF['paying_mon_fee' + mon],fill_value = 0)
    ##已还本金+待还本金
    result['paying_capital' + mon] = p6m_DF['payed_capital' + mon] + p6m_DF['bal' + mon]
    ##待还本金比率
    result['dh_balPercent' + mon] = p6m_DF['bal' + mon].divide(result['paying_capital' + mon],fill_value = 0)

    """月待还款额 = 月额度 - 已还款额 - 剩余可用额度"""
    result['dh_cpt_pymt' + mon] = p6m_DF['credit_limit' + mon] - p6m_DF['cpt_pymt' + mon] - p6m_DF['fopen_to_buy' + mon]
    ##月待还款比例
    result['dh_cpt_pymtPercent' + mon] = result['dh_cpt_pymt' + mon].divide( (p6m_DF['credit_limit' + mon] - p6m_DF['fopen_to_buy' + mon]),fill_value = 0)

    """月新建分期类比例 od_zdfq_brw"""
    result['od_zdfq_brwPercent'+mon] = p6m_DF['od_zdfq_brw'+mon] / p6m_DF['od_brw'+mon]
    """已还分期类月服务费所占比率"""
    result['payed_zdfq_mon_feePercent'+mon] = p6m_DF['payed_zdfq_mon_fee'+mon] / p6m_DF['payed_mon_fee'+mon]

for feature in ['dh_tot_fee','dh_tot_feePercent','dh_mon_fee','dh_mon_feePercent',
                'paying_capital','dh_balPercent','dh_cpt_pymt','dh_cpt_pymtPercent']:
    result['avg' + feature] = (result[feature + 'JUL16'] + result[feature + 'OCT16'] + result[feature + 'AUG16'] + \
                               result[feature + 'SEP16'] + result[feature + 'JUN16'] + result[feature + 'MAY16']) / 6.0

##计算用户MAY16（2016年5月），JUN16，JUL16，AUG16，SEP16应还完的订单数
result['paying_complete_od_cnt_InMAY16'] = p6m_DF['paying_complete_od_cntJUN16'] - p6m_DF['paying_complete_od_cntMAY16']
result['paying_complete_od_cnt_InJUN16'] = p6m_DF['paying_complete_od_cntJUL16'] - p6m_DF['paying_complete_od_cntJUN16']
result['paying_complete_od_cnt_InJUL16'] = p6m_DF['paying_complete_od_cntAUG16'] - p6m_DF['paying_complete_od_cntJUL16']
result['paying_complete_od_cnt_InAUG16'] = p6m_DF['paying_complete_od_cntSEP16'] - p6m_DF['paying_complete_od_cntAUG16']
result['paying_complete_od_cnt_InSEP16'] = p6m_DF['paying_complete_od_cntOCT16'] - p6m_DF['paying_complete_od_cntSEP16']

##计算用户MAY16，JUN16，JUL16，AUG16，SEP16已还完的订单比率
result['payed_complete_od_cntMAY16Percent'] = p6m_DF['payed_complete_od_cntMAY16'].divide(result['paying_complete_od_cnt_InMAY16'],fill_value = 0)
result['payed_complete_od_cntJUN16Percent'] = p6m_DF['payed_complete_od_cntJUN16'].divide(result['paying_complete_od_cnt_InJUN16'],fill_value = 0)
result['payed_complete_od_cntJUL16Percent'] = p6m_DF['payed_complete_od_cntJUL16'].divide(result['paying_complete_od_cnt_InJUL16'],fill_value = 0)
result['payed_complete_od_cntAUG16Percent'] = p6m_DF['payed_complete_od_cntAUG16'].divide(result['paying_complete_od_cnt_InAUG16'],fill_value = 0)
result['payed_complete_od_cntSEP16Percent'] = p6m_DF['payed_complete_od_cntSEP16'].divide(result['paying_complete_od_cnt_InSEP16'],fill_value = 0)

##计算用户MAY16，JUN16，JUL16，AUG16，SEP16提前还完的订单数
result['acre_repay_od_cntInMAY16'] = p6m_DF['acre_repay_od_cntJUN16'] - p6m_DF['acre_repay_od_cntMAY16']
result['acre_repay_od_cntInJUN16'] = p6m_DF['acre_repay_od_cntJUL16'] - p6m_DF['acre_repay_od_cntJUN16']
result['acre_repay_od_cntInJUL16'] = p6m_DF['acre_repay_od_cntAUG16'] - p6m_DF['acre_repay_od_cntJUL16']
result['acre_repay_od_cntInAUG16'] = p6m_DF['acre_repay_od_cntSEP16'] - p6m_DF['acre_repay_od_cntAUG16']
result['acre_repay_od_cntInSEP16'] = p6m_DF['acre_repay_od_cntOCT16'] - p6m_DF['acre_repay_od_cntSEP16']

##计算用户MAY16，JUN16，JUL16，AUG16，SEP16提前还完的订单比率
result['acre_repay_od_cntInMAY16Percent'] = result['acre_repay_od_cntInMAY16'].divide(result['paying_complete_od_cnt_InMAY16'],fill_value = 0)
result['acre_repay_od_cntInJUN16Percent'] = result['acre_repay_od_cntInJUN16'].divide(result['paying_complete_od_cnt_InJUN16'],fill_value = 0)
result['acre_repay_od_cntInJUL16Percent'] = result['acre_repay_od_cntInJUL16'].divide(result['paying_complete_od_cnt_InJUL16'],fill_value = 0)
result['acre_repay_od_cntInAUG16Percent'] = result['acre_repay_od_cntInAUG16'].divide(result['paying_complete_od_cnt_InAUG16'],fill_value = 0)
result['acre_repay_od_cntInSEP16Percent'] = result['acre_repay_od_cntInSEP16'].divide(result['paying_complete_od_cnt_InSEP16'],fill_value = 0)

##计算用户MAY16，JUN16，JUL16，AUG16，SEP16应还完的本金金额
result['paying_complete_od_brw_InMAY16'] = p6m_DF['paying_complete_od_brwJUN16'] - p6m_DF['paying_complete_od_brwMAY16']
result['paying_complete_od_brw_InJUN16'] = p6m_DF['paying_complete_od_brwJUL16'] - p6m_DF['paying_complete_od_brwJUN16']
result['paying_complete_od_brw_InJUL16'] = p6m_DF['paying_complete_od_brwAUG16'] - p6m_DF['paying_complete_od_brwJUL16']
result['paying_complete_od_brw_InAUG16'] = p6m_DF['paying_complete_od_brwSEP16'] - p6m_DF['paying_complete_od_brwAUG16']
result['paying_complete_od_brw_InSEP16'] = p6m_DF['paying_complete_od_brwOCT16'] - p6m_DF['paying_complete_od_brwSEP16']

##计算用户MAY16，JUN16，JUL16，AUG16，SEP16已还完的本金金额比率
result['payed_complete_od_brwMAY16Percent'] = p6m_DF['paying_complete_od_brwMAY16'].divide(result['paying_complete_od_brw_InMAY16'],fill_value = 0)
result['payed_complete_od_brwJUN16Percent'] = p6m_DF['paying_complete_od_brwJUN16'].divide(result['paying_complete_od_brw_InJUN16'],fill_value = 0)
result['payed_complete_od_brwJUL16Percent'] = p6m_DF['paying_complete_od_brwJUL16'].divide(result['paying_complete_od_brw_InJUL16'],fill_value = 0)
result['payed_complete_od_brwAUG16Percent'] = p6m_DF['paying_complete_od_brwAUG16'].divide(result['paying_complete_od_brw_InAUG16'],fill_value = 0)
result['payed_complete_od_brwSEP16Percent'] = p6m_DF['paying_complete_od_brwSEP16'].divide(result['paying_complete_od_brw_InSEP16'],fill_value = 0)

##计算用户MAY16，JUN16，JUL16，AUG16，SEP16提前还款的本金金额
result['acre_repay_od_cptInMAY16'] = p6m_DF['acre_repay_od_cptJUN16'] - p6m_DF['acre_repay_od_cptMAY16']
result['acre_repay_od_cptInJUN16'] = p6m_DF['acre_repay_od_cptJUL16'] - p6m_DF['acre_repay_od_cptJUN16']
result['acre_repay_od_cptInJUL16'] = p6m_DF['acre_repay_od_cptAUG16'] - p6m_DF['acre_repay_od_cptJUL16']
result['acre_repay_od_cptInAUG16'] = p6m_DF['acre_repay_od_cptSEP16'] - p6m_DF['acre_repay_od_cptAUG16']
result['acre_repay_od_cptInSEP16'] = p6m_DF['acre_repay_od_cptOCT16'] - p6m_DF['acre_repay_od_cptSEP16']

##计算用户MAY16，JUN16，JUL16，AUG16，SEP16提前还款的本金金额比率
result['acre_repay_od_cptInMAY16Percent'] = result['acre_repay_od_cptInMAY16'].divide(result['paying_complete_od_brw_InMAY16'],fill_value = 0)
result['acre_repay_od_cptInJUN16Percent'] = result['acre_repay_od_cptInJUN16'].divide(result['paying_complete_od_brw_InJUN16'],fill_value = 0)
result['acre_repay_od_cptInJUL16Percent'] = result['acre_repay_od_cptInJUL16'].divide(result['paying_complete_od_brw_InJUL16'],fill_value = 0)
result['acre_repay_od_cptInAUG16Percent'] = result['acre_repay_od_cptInAUG16'].divide(result['paying_complete_od_brw_InAUG16'],fill_value = 0)
result['acre_repay_od_cptInSEP16Percent'] = result['acre_repay_od_cptInSEP16'].divide(result['paying_complete_od_brw_InSEP16'],fill_value = 0)

##6个月逾期的比率
result['allfoverdue_paying_day'] = p6m_DF['foverdue_paying_dayMAY16']
result['allfoverdue_paying_cyc'] = p6m_DF['foverdue_paying_cycMAY16']
for mon in ['JUL16','OCT16','AUG16','SEP16','JUN16']:
    result['allfoverdue_paying_day'] = result['allfoverdue_paying_day'] + p6m_DF['foverdue_paying_day' + mon]
    result['allfoverdue_paying_cyc'] = result['allfoverdue_paying_cyc'] + p6m_DF['foverdue_paying_cyc' + mon]

result['foverdue_paying_dayPercent'] = result['allfoverdue_paying_day'].divide(p6m_DF['foverdue_payed_dayMAY16'],fill_value = 0)
##过去6个月的逾期天数 / 5月的用户历史逾期天数
result['foverdue_paying_cycPercent'] = result['allfoverdue_paying_cyc'].divide(p6m_DF['foverdue_payed_cycMAY16'],fill_value = 0)

##额度是否增加
result['credit_limit'] = p6m_DF['credit_limitOCT16'] - p6m_DF['credit_limitMAY16']

result = result.replace(-np.inf,0)
result = result.replace(np.inf,0)
result = result.fillna(0)

print result.shape

result.iloc[:50000,:].to_csv('../train_file/p6m挖掘出的新特征(删除部分特征).csv', index=False)
result.iloc[50000:,:].to_csv('../test_file/p6m挖掘出的新特征(删除部分特征).csv', index=False)

