#coding=utf-8

import pandas as pd
import numpy as np

loginSecene_DF1 = pd.read_csv('../train_file/特征处理后的login_scene_mdl.csv')
loginSecene_DF2 = pd.read_csv('../test_file/经过特征处理后的login_scene_offtime.csv')
loginSecene_DF = pd.concat([loginSecene_DF1,loginSecene_DF2],axis=0)

features = pd.read_csv('../train_file/login_scene_mdl.csv').columns



# features = pd.read_csv('../test_file/login_scene_offtime.csv').columns

cols = loginSecene_DF.columns.tolist()

StringFeatureLs = ['c_scene_reg_max_dur','c_scene_dl_max_dur','c_scene_od_max_dur','c_scene_rp_max_dur',
'c_scene_xgxx_max_dur','c_scene_plsp_max_dur','c_scene_sczl_max_dur','c_scene_sh_max_dur',
                   'c_scene_reg_min_dur','c_scene_dl_min_dur','c_scene_od_min_dur','c_scene_rp_min_dur',
'c_scene_xgxx_min_dur','c_scene_plsp_min_dur','c_scene_sczl_min_dur','c_scene_sh_min_dur',
                   'c_scene_log_max_dur','c_scene_log_min_dur']

featureLs = ['c_scene_reg_tot_cnt'
,'c_scene_dl_tot_cnt'
,'c_scene_od_tot_cnt'
,'c_scene_rp_tot_cnt'
,'c_scene_xgxx_tot_cnt'
,'c_scene_plsp_tot_cnt'
,'c_scene_sczl_tot_cnt'
,'c_scene_sh_tot_cnt']

##场景总次数
for mon in ['JUL16','SEP16', 'AUG16', 'OCT16', 'JUN16', 'MAY16']:
    loginSecene_DF['sum_scene'+mon] = loginSecene_DF['c_scene_reg_tot_cnt'+mon]
    cols.remove('c_scene_reg_tot_cnt'+mon)
    for fea in featureLs[1:]:
        loginSecene_DF['sum_scene'+mon] += loginSecene_DF[fea+mon]
        cols.remove(fea+mon)
    for fea in featureLs:
        loginSecene_DF[fea+'Percent'+mon] = loginSecene_DF[fea+mon].divide(loginSecene_DF['sum_scene'+mon],fill_value = 0)

featureLs = ['c_scene_pc_tot_cnt','c_scene_app_tot_cnt'
,'c_scene_h5_tot_cnt'
,'c_scene_android_tot_cnt'
,'c_scene_ios_tot_cnt'
,'c_scene_log_avg_dur']

##使用工具浏览场景的总次数
for mon in ['JUL16','SEP16', 'AUG16', 'OCT16', 'JUN16', 'MAY16']:
    loginSecene_DF['sum_sceneBy'+mon] = loginSecene_DF['c_scene_pc_tot_cnt'+mon]
    cols.remove('c_scene_pc_tot_cnt' + mon)
    for fea in featureLs[1:]:
        loginSecene_DF['sum_sceneBy'+mon] +=loginSecene_DF[fea+mon]
        cols.remove(fea + mon)
    for fea in featureLs:
        loginSecene_DF[fea+'Percent'+mon] = loginSecene_DF[fea+mon].divide(loginSecene_DF['sum_sceneBy'+mon],fill_value = 0)


for val in features:
    if val in ['fuid_md5','pyear_month','cyc_date']:
        continue
    for subval in ['JUL16','SEP16', 'AUG16', 'OCT16', 'JUN16', 'MAY16']:
        loginSecene_DF[val + subval].astype(float)
    sum = loginSecene_DF[val + 'JUN16']
    for fea in cols:

        if fea[:-5] == val and fea != (val+'JUN16'):
            sum+=loginSecene_DF[fea]
    loginSecene_DF['avg' + val] = sum / 6.0

loginSecene_DF = loginSecene_DF.replace(np.inf,0)
loginSecene_DF = loginSecene_DF.replace(-np.inf,0)
loginSecene_DF = loginSecene_DF.fillna(0)

loginSecene_DF.iloc[:50000,:].to_csv('../train_file/求平均值后login_scene_mdl.csv', index=False)
loginSecene_DF.iloc[50000:,:].to_csv('../test_file/求平均值后login_scene_offtime.csv', index=False)






