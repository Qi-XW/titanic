import numpy as np
import pandas as pd
datatrain = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

datatrain = datatrain.drop(labels=['PassengerId','Name','SibSp','Parch','Ticket', 'Fare','Cabin', 'Embarked'], axis=1)

datatrain = datatrain.fillna(datatrain.mean()['Age'])
datatrain_dummy = pd.get_dummies(datatrain[['Sex']])
datatrain_conti = pd.DataFrame(datatrain, columns=['Survived', 'Pclass', 'Age'],
                               index=datatrain.index)
datatrain = datatrain_conti.join(datatrain_dummy)

datatrain['cate_age'] = pd.cut(datatrain['Age'],[0,15,30,60,80])
datatrain.loc[datatrain['Age']<=15,'Age'] = 0
datatrain.loc[(datatrain['Age']>15)&(datatrain['Age']<=30),'Age'] = 1
datatrain.loc[(datatrain['Age']>30)&(datatrain['Age']<=60),'Age'] = 2
datatrain.loc[datatrain['Age']>60,'Age'] = 3
datatrain['Age'].astype(int)
datatrain = datatrain.drop('cate_age',axis=1)

train = datatrain.values
gnd_T = train[:,0]

def naive_bayes(x,gnd):
    x = x.T
    (a,b) = x.shape
    my_lab = set(gnd)
    houyan_0 = np.zeros((4,4))
    houyan_1 = np.zeros((4, 4))
    pw = []
    for i in my_lab:
        pw.append(np.sum(gnd == i))
        pw[int(i)] = pw[int(i)]/len(gnd)

    for i in my_lab:
        for j in range(a-1):
            hou_test = set(x[j+1])
            for l in hou_test:
                arr=x[j+1].copy()
                (k,) = arr[np.logical_and(gnd == i,x[j+1] == l)].shape
                p_jl = k/(pw[int(i)]*len(gnd))
                if i == 0:
                    houyan_0[int(j),int(l)] = p_jl
                else:
                    houyan_1[int(j),int(l)] = p_jl

    return pw,houyan_0,houyan_1

(pw,houyan_0,houyan_1) = naive_bayes(train,gnd_T)

datatest = data_test.drop(labels=['PassengerId','Name','SibSp','Parch','Ticket', 'Fare','Cabin', 'Embarked'], axis=1)
datatest = datatest.fillna(datatest.mean()['Age'])
datatest_dummy = pd.get_dummies(datatest['Sex'])
datatest_conti = pd.DataFrame(datatest, columns=['Pclass', 'Age'], index=datatest.index)
datatest = datatest_conti.join(datatest_dummy)

datatest['cate_age'] = pd.cut(datatest['Age'],[0,15,30,60,80])
datatest.loc[datatest['Age']<=15,'Age'] = 0
datatest.loc[(datatest['Age']>15)&(datatest['Age']<=30),'Age'] = 1
datatest.loc[(datatest['Age']>30)&(datatest['Age']<=60),'Age'] = 2
datatest.loc[datatest['Age']>60,'Age'] = 3
datatest['Age'].astype(int)
datatest = datatest.drop('cate_age',axis=1)

test = datatest.values
(m1,m2) = test.shape
sur = []

for i in range(m1):
    sur_1 = 0
    sur_0 = 0
    su0 = 1
    su1 = 1
    for j in range(m2):
        su0 *= houyan_0[j][int(test[int(i)][j])]
    sur_0 = su0*pw[0]
    for j in range(m2):
        su1 *= houyan_1[j][int(test[int(i)][j])]
    sur_1 = su1*pw[1]
    if sur_0 > sur_1:
        sur.append(0)
    else:
        sur.append(1)

passenger = []
for i in range(len(sur)):
    passenger.append(i+1)
df_sur = pd.DataFrame(passenger,columns=['PassengerId'])
df_sur['Survived'] = sur
df_sur.to_csv('submission.csv',index=False)