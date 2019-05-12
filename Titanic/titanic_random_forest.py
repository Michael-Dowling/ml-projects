import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

train = pd.read_csv('train.csv')

train.info()

train.drop(['PassengerId','Ticket'],axis=1,inplace=True)


#plt.subplots(figsize=(9,5))
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False)



#so many null values in cabin
train.drop('Cabin',axis=1,inplace=True)

#exploring the data
#plt.subplots(figsize=(9,5))
#sns.countplot(x='Survived',hue='Sex',data=train)

sns.catplot(x="Pclass", col="Survived",data=train,kind="count")

plt.subplots(figsize=(9,5))

ax=sns.heatmap(train[["Survived","Pclass","Age","SibSp","Parch","Fare"]].corr(),annot=True,fmt=".2f",cmap="Blues")

def insert_age(cols):
    age = cols[0]
    Pclass = cols[1]
    if pd.isnull(age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    
    return age

train['Age'] = train[['Age','Pclass']].apply(insert_age,axis=1)

plt.subplots()

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="YlGnBu_r")



train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)

#fix fare skewness
train["Fare"] = train["Fare"].map(lambda i : np.log(i) if i > 0 else 0)

plt.subplots()

sns.distplot(train["Fare"],bins=60,color="g")


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train.drop('SibSp',axis=1,inplace=True)
train.drop('Parch',axis=1,inplace=True)

train.head()

#feature engineering - taking the name and changing it to only have the title
#names are given in the format Lastname, Ttl. Firstname (and any additional middle names)
#so get title after the comma (", ") and before the period

titles = [i.split(", ")[1].split(".")[0].strip() for i in train["Name"]]
train["Title"] = pd.Series(titles)

#convert titles to be one of Mr, Mrs, Miss or Rare (Rare used for other titles such as Dr or Sir)
train["Title"] = train["Title"].replace(['Lady', 'the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
train["Title"] = train["Title"].replace(['Mlle','Miss'])
train["Title"] = train["Title"].replace('Ms','Miss')
train["Title"] = train["Title"].replace('Mme','Mrs')
#Adding columns for each title (1 if they have that title, 0 if not)
title = pd.get_dummies(train['Title'],drop_first=True)
train = pd.concat([train,title],axis=1)

#dropping title and name
train.drop(['Name','Title'],axis=1,inplace=True)

#change categorical variables to numerical ones - for embarked and sex
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Embarked'],axis=1,inplace=True)

#building the model
x = train.drop('Survived',axis=1)
y = train['Survived']

#split into train/test splits
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


#random forest classification model
from sklearn.ensemble import RandomForestClassifier

rfmodel = RandomForestClassifier(random_state=0,n_estimators=450,criterion='gini',n_jobs=-1,max_depth=8, min_samples_leaf=1, min_samples_split=11)

rfmodel.fit(x_train,y_train)    #train the model with input x_train and ground truth y_train

#output predictions on train part
predictions = rfmodel.predict(x_test)

#check results

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))