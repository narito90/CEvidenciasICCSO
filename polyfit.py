import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd


dataset = genfromtxt('exam_B_dataset.csv', delimiter=',')
dataset = pd.read_csv('exam_B_dataset.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

print()

def polyfit1(x,y,n):

    def inv(A):
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    A=np.hstack([c1,xT])
    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))

""" xB=[8450,9600,11250,9550,14260,14115,10084,10382,6120,7420,11200,11924,12968,10652,10920,6120,11241,10791,13695,7560,14215,7449,9742,4224,8246,14230,7200,11478,16321,6324,8500,8544,11049,10552,7313,13418,10859,8532,7922,6040,8658,16905,9180,9200,7945,7658,12822,11096,4456,7742]
yB=[169277.0525,187758.394,183583.6836,179317.4775,150730.08,177150.9892,172070.6892,175110.9565,162011.6988,160726.2478,157933.2795,145291.245,159672.0176,164167.5183,150891.6382,179460.9652,185034.6289,182352.1926,183053.4582,187823.3393,186544.1146,158230.7752,190552.8293,147183.6749,185855.3009,174350.4707,201740.6207,162986.3789,162330.1991,165845.9386,180929.6229,163481.5015,187798.0767,198822.1989,194868.4099,152605.2986,147797.7028,150521.969,146991.6302,150306.3078,141164.3725,151133.707,156214.0425,171992.7607,173214.9125,192429.1873,190878.6951,194542.5441,191849.4391,176363.7739]
x=[1,1.6,3.4,4,5.2]
y=[1.2,2,2.4,3.5,3.5]
"""

polyfit1(x,y,1)

def polyfit2(x,y,n):

    def inv(A):
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    A=np.hstack([c1,c2,c3])
    print(A)
    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))

"""
polyfit2(x,y,2)
"""

polyfit1(x,y,1)
