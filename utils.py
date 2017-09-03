#coding=utf-8

import pandas as pd
import numpy as np

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