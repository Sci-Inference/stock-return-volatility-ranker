import numpy as np
import pandas as pd
from itertools import combinations




def score_to_rank(symbolList:list,indexList:list,prd:list):
    res = {}
    for i in symbolList:
        res[i] = 0
    for ix,p in zip(indexList,prd):
        res[ix[0]] += p.reshape(-1)
        res[ix[1]] += 1 - p.reshape(-1)
    return res