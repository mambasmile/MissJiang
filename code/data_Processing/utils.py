#coding=utf-8

import pandas as pd
import numpy as np
import csv
from testFileConfig import *
from trainFileConfig import *
import json
import xlrd

class utils:

    @staticmethod
    def rowTransform(dataFrame,cyc_date,rowIndex):
        index = dataFrame.columns
        length=dataFrame.shape[0]
        cols=dataFrame.shape[1]
        all_index = []
        data_Dict = {}
        data_Dict['fuid_md5'] = dataFrame.iloc[0][0]
        all_index.append('fuid_md5')
        for val in index[1:]:
            for i in xrange(0,length):
                date = "".join(cyc_date[i].split('-')[0:2])
                tmpval = val + date

                all_index.append(tmpval)
        # print all_index.__len__()

        ls_index=1
        for j in xrange(1,cols):
            for i in xrange(0,length):
                data_Dict[all_index[ls_index]] = dataFrame.iloc[i][j]
                ls_index+=1
        return pd.DataFrame(data_Dict,columns=all_index,index=[rowIndex])

    @staticmethod
    def processP6M(dataFrame,rowName):
        index = 1
        first_origin_dataFrame = dataFrame[dataFrame['fuid_md5'] == rowName[0]]
        dataFrame = dataFrame.drop(first_origin_dataFrame.index.values,axis=0)
        cyc_date = first_origin_dataFrame['cyc_date'].values
        allDataFrame = utils.rowTransform(first_origin_dataFrame,cyc_date,0)
        # print allDataFrame
        for uid in rowName[1:]:
            tmpDataFrame = dataFrame[dataFrame['fuid_md5'] == uid]
            dataFrame = dataFrame.drop(tmpDataFrame.index.values, axis=0)
            res_DataFrame = utils.rowTransform(tmpDataFrame,cyc_date,index)
            allDataFrame = pd.concat((allDataFrame,res_DataFrame),axis=0)
            index+=1
        return allDataFrame

    @staticmethod
    def dataFrameLog1P(series):
        length = series.shape[0]
        # # print length
        # for i in xrange(length):
        #     if series[i] != 0:
        #         series[i] = np.log1p(float(series[i]))

        return np.log1p(series)

    @staticmethod
    def transformDataFrame(file,outfile):
        tmpSet = set()
        tmpSet.add('fuid_md5')
        dataDict = {}
        csv_reader = csv.reader(open(file))
        flag = 0
        # count = 0
        for row in csv_reader:
            # count+=1
            # if count==26:
            #     break
            if flag == 0:
                columns = row
                flag+=1
            else:
                if row[0] not in dataDict:
                    dataDict[row[0]] = {}
                tmpKey = row[1][2:7]
                for i in range(3,columns.__len__()):
                    dataDict[row[0]][columns[i]+tmpKey]=row[i]
                    tmpSet.add(columns[i]+tmpKey)

        # print dataDict

        resDict = {}
        columnsName = list(tmpSet)
        # print columnsName
        # columnsName.extend(dataDict['1f9aa4769d4cee1656a17c5546b95839'].keys())
        for val in columnsName:
            resDict[val]=[]
        for keyStr in dataDict:
            resDict['fuid_md5'].append(keyStr)
            for tmpKey in dataDict[keyStr]:
                # if tmpKey not in resDict:
                #     resDict[tmpKey]=[]
                resDict[tmpKey].append(dataDict[keyStr][tmpKey])
        # print resDict
        # for tmpDict in dataDict:
        #     for keyStr in tmpDict:
        #         resDict[keyStr].append(tmpDict[keyStr])
        # allColumns = resDict.keys()
        # for val in columns:
        #     for tmpVal in allColumns:
        #         if tmpVal[0:-6] == val:
        #             columnsName.append(tmpVal)
        print resDict.__len__()

        # dataFrame = pd.DataFrame(resDict,columns=columnsName)
        # dataFrame.to_csv(outfile,index=False)

    @staticmethod
    def reName6Month(file,outfile):
        p6_DF = pd.read_csv(file)
        columnsName = p6_DF.columns
        res_columns = []
        tmpLs = []
        for val in columnsName:
            tmpLs.append(val)
            if val == 'fuid_md5':
                res_columns.append(val)
            else:
                tmpStr = val[-5:]
                if tmpStr == 'JUL16':
                    reStr = 'MAY16'
                elif tmpStr == 'AUG16':
                    reStr = 'JUN16'
                elif tmpStr == 'SEP16':
                    reStr = 'JUL16'
                elif tmpStr == 'OCT16':
                    reStr = 'AUG16'
                elif tmpStr == 'NOV16':
                    reStr = 'SEP16'
                else:
                    reStr = 'OCT16'
                restr = val[0:-5] + reStr
                res_columns.append(restr)
        # print tmpLs
        # print res_columns
        # p6_DF.rename(res_columns,inplace=True)
        p6_DF.columns = res_columns
        p6_DF.to_csv(outfile,index=False)

    """将测试集合中的时间减少两个月份"""
    @staticmethod
    def reName12Month(file, outfile):
        p12_DF = pd.read_csv(file)
        columnsName = p12_DF.columns
        res_columns = []
        tmpLs = []
        for val in columnsName:
            tmpLs.append(val)
            if val == 'fuid_md5':
                res_columns.append(val)
            else:
                tmpStr = val[-5:]
                if tmpStr == 'JAN16':
                    reStr = 'NOV15'
                elif tmpStr == 'FEB16':
                    reStr = 'DEC15'
                elif tmpStr == 'MAR16':
                    reStr = 'JAN16'
                elif tmpStr == 'APR16':
                    reStr = 'FEB16'
                elif tmpStr == 'MAY16':
                    reStr = 'MAR16'
                elif tmpStr == 'JUN16':
                    reStr = 'APR16'
                elif tmpStr == 'JUL16':
                    reStr = 'MAY16'
                elif tmpStr == 'AUG16':
                    reStr = 'JUN16'
                elif tmpStr == 'SEP16':
                    reStr = 'JUL16'
                elif tmpStr == 'OCT16':
                    reStr = 'AUG16'
                elif tmpStr == 'NOV16':
                    reStr = 'SEP16'
                elif tmpStr == 'DEC16':
                    reStr = 'OCT16'
                restr = val[0:-5] + reStr
                res_columns.append(restr)
        # print tmpLs
        # print res_columns
        # p6_DF.rename(res_columns,inplace=True)
        p12_DF.columns = res_columns
        p12_DF.to_csv(outfile, index=False)

    @staticmethod
    def writeFreatureToText(featureLs,outfile):
        with open(outfile,'a') as e:
            for val in featureLs:
                e.write(val+'\n')

    """将特征出现次数字典以及特征名写入文件"""
    @staticmethod
    def writeFeatureDetailTofile(featureName,dataFrame,outfile):
        with open(outfile, 'a') as e:
            e.write('-------------'+featureName+'\n')
            columnsName = dataFrame.columns
            for val in columnsName:
                if val[:-5] == featureName:
                    dataDict = dataFrame[val].value_counts().to_dict()
                    e.write('\n'+val+'\n')
                    for key in dataDict:
                        e.write(str(key) + ":" + str(dataDict[key]) + '\n')

    """对特征进行标准化"""
    @staticmethod
    def standardFeature(dataFrame,dataInfoFrame,featurefile,outpath):
        columnsLs = dataFrame.columns
        featureLs = []
        resFeature = []
        with open(featurefile,'r') as e:
            for line in e:
                ele = line.strip()
                featureLs.append(ele)
        for cols in columnsLs:
            if cols[0:-5] in featureLs:
                resFeature.append(cols)
        for feature in resFeature:
            mean = dataInfoFrame.ix['mean'][feature]
            std = dataInfoFrame.ix['std'][feature]
            dataFrame[feature] = (dataFrame[feature] - mean) / std
        dataFrame.to_csv(outpath,index=False)

    """对特征进行标准化"""
    @staticmethod
    def log1pFeature(dataFrame, featureLs, outpath):
        for val in ['fcredit_update_timeSEP16','fcredit_update_timeJUL16',
                       'fcredit_update_timeMAY16','fcredit_update_timeOCT16'
                        ,'fcredit_update_timeAUG16','fcredit_update_timeJUN16']:
            dataFrame.pop(val)
            featureLs.remove(val)
        for feature in featureLs:
            if feature != 'fuid_md5':
                dataFrame[feature] = np.log1p(dataFrame[feature])
        dataFrame.to_csv(outpath,index=False)

    """对特征进行标准化"""

    """观察特征集合中是否有交集
    featureFile1:重要的文件
    featureFile2:非重要的文件
    """
    @staticmethod
    def checkFeatureLs(featureFile1,featureFile2):
        tmpLs = [] ##重要的
        tmpLs1 = [] ##非重要的
        with open(featureFile1,'r') as e:
            for line in e:
                ele = line.strip()
                tmpLs.append(ele)
        with open(featureFile2,'r') as e:
            for line in e:
                ele = line.strip()
                tmpLs1.append(ele)
        print [val for val in tmpLs1 if val in tmpLs]

    """
    求剩余的特征集合
    featureFile1 表示值存在负数的特征的文件
    featureFile2 表示值为种类的特征的文件
    """
    @staticmethod
    def subFeatures(dataFrame,featureFile1,featureFile2):
        columnsLs = dataFrame.columns.tolist()
        tmpLs = []
        tmpLs1 = []
        resLs = []
        with open(featureFile1, 'r') as e:
            for line in e:
                ele = line.strip()
                tmpLs.append(ele)
        with open(featureFile2, 'r') as e:
            for line in e:
                ele = line.strip()
                tmpLs1.append(ele)
        for val in columnsLs:
            if val[0:-5] not in tmpLs1 and val[0:-5] not in tmpLs:
                resLs.append(val)
        return resLs

    @staticmethod
    def deleteFeature(dataFrame,former,later,midLs,months):
        for val in midLs:
            tmpVal = former + val + later
            for col in months:
                dataFrame.pop(tmpVal + col)
        return dataFrame

    """读取字典"""
    @staticmethod
    def readDictExcel(dictfile,lsfile):
        rd = xlrd.open_workbook(dictfile)
        sheet1= rd.sheets()[1]
        featureDict = {}
        for i in xrange(sheet1.nrows):
            featureDict[sheet1.cell(i,0).value] = sheet1.cell(i,2).value

        with open(lsfile,'w') as e:
            for key in featureDict:
                if key != '':
                    e.write(key.encode('utf-8')+":"+featureDict[key].encode('utf-8')+'\n')

    """查找特征"""
    @staticmethod
    def findFeatureName(lsfile,featureKey):
        with open(lsFile, 'r') as e:
            for line in e:
                data = line.strip().split(":")
                if data[0] in featureKey:
                    print data[1]

    """对特征列表进行重构"""
    @staticmethod
    def rebuildFeatureLs(featureLs):
        resLs = set()
        for val in featureLs:
            resLs.add(val[0:-5])
        return list(resLs)



