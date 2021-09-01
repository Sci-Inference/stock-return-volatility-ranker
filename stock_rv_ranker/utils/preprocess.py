import numpy as np
import pandas as pd
from itertools import combinations


def rank_to_score_sequence(data:dict,symbolList:list,featureList:list,targetCol:str):
    xi = []
    xj = []
    y = []
    indexList = []
    for i in list(combinations(symbolList,2)):
        xi.append(np.stack([np.array(data[i[0]][j]) for j in featureList],axis =-1))
        xj.append(np.stack(([np.array(data[i[1]][j]) for j in featureList]),axis =-1))
        if targetCol is not None:
            y.append((1 if data[i[0]][targetCol] > data[i[1]][targetCol] else 0))
        indexList.append(i)
    return np.stack(xi,0),np.stack(xj,0),np.array(y),indexList



def rank_to_score(data:dict,symbolList:list,featureList:list,targetCol:str):
    xi = []
    xj = []
    y = []
    indexList = []
    for i in list(combinations(symbolList,2)):
        xi.append([data[i[0]][j] for j in featureList])
        xj.append([data[i[1]][j] for j in featureList])
        if targetCol is not None:
            y.append(1 if data[i[0]][targetCol] > data[i[1]][targetCol] else 0)
        indexList.append(i)
    return np.array(xi),np.array(xj),np.array(y),indexList


class Batch_Dataset(object):
    def __init__(self) -> None:
        super().__init__()
        self.indexList=[]


    def create_numpy_dataset(self,data:pd.DataFrame,featureList:list,DateCol:str = 'Date',indexCol:str = 'symbol',targetCol:str='profit'):
        dateList = data[DateCol].unique()
        xi = []
        xj = []
        y = []
        indexList = []
        for i in dateList:
            tdd = data[data['Date'] == i].set_index(indexCol)
            query_xi,query_xj,query_y,query_lid = rank_to_score(tdd.to_dict('index'),tdd.index.unique(),featureList,targetCol)
            xi.extend(query_xi)
            xj.extend(query_xj)
            y.extend(query_y)
            indexList.extend(query_lid)
        return np.array(xi),np.array(xj),np.array(y),indexList


    def create_sequence_numpy_dataset(self,data:pd.DataFrame,featureList:list,DateCol:str = 'Date',indexCol:str = 'symbol',targetCol='profit'):
        dateList = data[DateCol].unique()
        xi = []
        xj = []
        y = []
        indexList = []
        for i in dateList:
            tdd = data[data['Date'] == i].set_index(indexCol)
            query_xi,query_xj,query_y,query_lid = rank_to_score_sequence(tdd.to_dict('index'),tdd.index.unique(),featureList,targetCol)
            xi.append(query_xi)
            xj.append(query_xj)
            y.extend(query_y)
            indexList.extend(query_lid)
        return np.concatenate(xi,0),np.concatenate(xj,0),np.array(y),indexList



    def create_sequence_numpy_generator(self,data:pd.DataFrame,featureList:list,DateCol:str = 'Date',indexCol:str = 'symbol',targetCol='profit'):
        dateList = data[DateCol].unique()
        for i in dateList:
            tdd = data[data['Date'] == i].set_index(indexCol)
            query_xi,query_xj,query_y,query_lid = rank_to_score_sequence(tdd.to_dict('index'),tdd.index.unique(),featureList,targetCol)
            yield query_xi,query_xj,query_y,query_lid