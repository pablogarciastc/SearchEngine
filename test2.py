import numpy as np
idealRecall=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
realRecall=[0.029411764705882353, 0.058823529411764705, 0.08823529411764706, 0.11764705882352941, 0.14705882352941177]
precision=[1.0, 0.4, 0.21428571428571427, 0.26666666666666666, 0.29411764705882354]
newPrecision=[0,0,0,0,0,0,0,0,0,0,0]
ideal=np.array(idealRecall)
real = np.array(realRecall)
precisionNump = np.array(precision)
for i in range(len(realRecall)):
    if 0 < realRecall[i] < 0.1:
        if newPrecision[1] < precision[i]:
            newPrecision[1] = precision[i]
    if 0.1 < realRecall[i] < 0.2:
        if newPrecision[2] < precision[i]:
            newPrecision[2] = precision[i]
    if 0.2 < realRecall[i] < 0.3:
        if newPrecision[3] < precision[i]:
            newPrecision[3] = precision[i]
    if 0.3 < realRecall[i] < 0.4:
        if newPrecision[4] < precision[i]:
            newPrecision[4] = precision[i]
    if 0.4 < realRecall[i] < 0.5:
        if newPrecision[5] < precision[i]:
            newPrecision[5] = precision[i]
    if 0.5 < realRecall[i] < 0.6:
        if newPrecision[6] < precision[i]:
            newPrecision[6] = precision[i]
    if 0.6 < realRecall[i] < 0.7:
        if newPrecision[7] < precision[i]:
            newPrecision[7] = precision[i]
    if 0.7 < realRecall[i] < 0.8:
        if newPrecision[8] < precision[i]:
            newPrecision[8] = precision[i]
    if 0.8 < realRecall[i] < 0.9:
        if newPrecision[9] < precision[i]:
            newPrecision[9] = precision[i]
    if 0.9 < realRecall[i] < 1:
        if newPrecision[10] < precision[i]:
            newPrecision[10] = precision[i]
print(newPrecision)






p = np.array(newPrecision)
m = np.zeros(p.size, dtype=bool)
precNorm = [0,0,0,0,0,0,0,0,0,0,0]
excptIndx=[]

for i in range(len(newPrecision)):
    m[excptIndx] = True
    a = np.ma.array(p, mask=m)
    precNorm[i]=newPrecision[np.argmax(a)]
    excptIndx.append(i)
print(precNorm)