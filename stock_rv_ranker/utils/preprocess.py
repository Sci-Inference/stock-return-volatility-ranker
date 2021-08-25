import numpy as np
import pandas as pd
from itertools import combinations



def rank_to_score(data,symbolList,featureList):
    xi = []
    xj = []
    y = []
    indexList = []
    for i in list(combinations(symbolList,2)):
        xi.append([data[i[0]][j] for j in featureList])
        xj.append([data[i[1]][j] for j in featureList])
        y.append(1 if data[i[0]]['profit'] > data[i[1]]['profit'] else 0)
        indexList.append(i)
    return xi,xj,y,indexList


def score_to_rank(symbolList,indexList,prd):
    res = {}
    for i in symbolList:
        res[i] = 0
    for ix,p in zip(indexList,prd):
        res[ix[0]] += p.reshape(-1)
        res[ix[1]] += 1 - p.reshape(-1)
    return pd.DataFrame.from_dict(res,'index').sort_values(0)

