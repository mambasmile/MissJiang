#coding=utf-8

"""
计算用户的账单日
"""

from datetime import datetime
import pandas as pd
from timeUtil import timeTransform
import json
from dateutil.relativedelta import relativedelta

# i = 0
# j=0
userComputeMoneyDay = {} ###用户的还款时间

# resList = []
tmpLs = []
with open(r'../train_file/求用户的账单日userComputeMoneyDay.csv','w') as e1:
    e1.write(",".join(['fuid_md5'
                          , 'userComputeMoneyDayNOV15', 'userComputeMoneyDayDEC15'
                          , 'userComputeMoneyDayJAN16', 'userComputeMoneyDayFEB16'
                          , 'userComputeMoneyDayMAR16', 'userComputeMoneyDayAPR16'
                          , 'userComputeMoneyDayJUN16', 'userComputeMoneyDayJUL16'
                          , 'userComputeMoneyDayAUG16', 'userComputeMoneyDaySEP16'
                          , 'userComputeMoneyDayOCT16', 'userComputeMoneyDayMAY16'
                       ]) + '\n')
    for i in xrange(50000):
        tmpLs = []
        with open(r'../train_file/1960时间处理的p12M_mdl.csv','r') as e:
            line = e.readlines()[i*12+1:i*12+13]
            for j in xrange(line.__len__()):
                data = line[j].strip().split(',')
                # if data[0] not in userComputeMoneyDay:
                #     userComputeMoneyDay[data[0]] = []
                if j==0:
                    tmpLs.append(data[0])
                    # j+=1
                tmpTime = datetime.strptime(data[2],'%Y-%m-%d')

                # tmpLs.append(str(pd.date_range(end=time, periods=21)[0]))
                tmpLs.append((tmpTime - relativedelta(days=+20)).strftime('%b-%d-%y %H:%M:%S'))

                # j+=1]
        e1.write(','.join(tmpLs)+'\n')
        e1.flush()
        # resList.append(tmpLs)

# print  resList

i=0
with open(r'../train_file/求用户的账单日userComputeMoneyDay.csv','r') as e1:
   for line in e1:
       if i==0:
           i+=1
           continue
       else:
           data = line.strip().split(',')
           userComputeMoneyDay[data[0]] = data[1:]

userOrderDict = {}
i = 0
with open(r'../train_file/od_in6m_mdl.csv','r') as e:
    for line in e:
        if i==0:
            i+=1
            continue
        else:
            data = line.strip().split(',')
            if data[0] not in userOrderDict:
                userOrderDict[data[0]] = {}
            time = datetime.strptime(data[2], '%d%b%y:%H:%M:%S')

            for timeDateIndex in range(userComputeMoneyDay[data[0]].__len__()):
                timeDate = userComputeMoneyDay[data[0]][timeDateIndex]
                datetimeType = datetime.strptime(timeDate,'%b-%d-%y %H:%M:%S')

                if time < datetimeType:
                    if timeDate not in userOrderDict[data[0]]:
                        userOrderDict[data[0]][timeDateIndex] = []

                    userOrderDict[data[0]][timeDateIndex].append([data[-4],data[-3],data[-2],data[-1]])
                    break

                if time > datetime.strptime(userComputeMoneyDay[data[0]][-1],'%b-%d-%y %H:%M:%S'):
                    userOrderDict[data[0]]['12'] = []
                    userOrderDict[data[0]]['12'].append([data[-4], data[-3], data[-2], data[-1]])
                    break
with open(r'../train_file/订单结果分析.txt','w') as e:
    for uid in userOrderDict:
        e.write(uid+','+json.dumps(userOrderDict[uid])+'\n')

df = pd.read_csv('../train_file/ud_mdl.csv')
fuidLs = df['fuid_md5'].tolist()

# fuidLs = ['1f9aa4769d4cee1656a17c5546b95839']
# userOrderDict = {'1f9aa4769d4cee1656a17c5546b95839':{6: [['301300', '0', '1', '15']], 7: [['20000', '0', '1', '3']], 8: [['20000', '0', '1', '1']], 9: [['30000', '0', '1', '3']], 10: [['26000', '0', '1', '3']], 11: [['1000', '0', '1', '1']]}}
fq_number = [] ##分期待还钱
avgfqdhPercent = [] ##分期待还比例
fqdh_money=[]

for uid in fuidLs:
    count = 0 ##分期数
    sumCount = 0 ##总分期数
    money = 0
    tmpfqPercent = []
    if uid in userOrderDict:
        for month in userOrderDict[uid]:
            for perOrder in userOrderDict[uid][month]:
                count = int(month)+int(perOrder[-1])-12
                try:
                    if count>0:
                        monthMoney = (int(perOrder[0]) - int(perOrder[1]))/float(perOrder[-1])
                        fqPercent = monthMoney / (int(perOrder[0]) - int(perOrder[1]))/float(int(perOrder[-1]))
                        fqPercent = count / float(int(perOrder[-1]))
                        tmpfqPercent.append(fqPercent)
                        sumCount+=count
                        money += monthMoney * count
                except ValueError,e:
                    print uid,userOrderDict[uid]
    fq_number.append(sumCount)
    fqdh_money.append(money)
    # print tmpfqPercent
    if tmpfqPercent.__len__() == 0:
        avgfqdhPercent.append(0.0)
    else:
        avgfqdhPercent.append(sum(tmpfqPercent)/float(tmpfqPercent.__len__()))

res_df = pd.DataFrame({'fuid_md5':fuidLs,
                       'orderfq_number':fq_number,
                       'fqdh_money':fqdh_money,
                       'orderAvgfqdhPercent':avgfqdhPercent})
res_df.to_csv('../train_file/用户订单信息.csv',index=False)
# print res_df





