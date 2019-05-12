import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


dataset = pd.read_csv('train.csv')

#drop the ticket and passesnger ID as those are simply unique identifiers
#drop cabin because it is not included for most passengers, and some have multiple cabins
dataset.drop(['Ticket','PassengerId','Cabin'],axis=1,inplace=True)

#print(dataset.describe(include='all'))

#deal with missing values

#replace missing ages with the mean age for that class
avg_age_by_class = [38,30,25]

dataset['Age'] = dataset[['Pclass','Age']].apply(lambda x: avg_age_by_class[int(x['Pclass']-1)] if pd.isnull(x['Age']) else x['Age'], axis=1)

#replace missing embarked ports with most common embarked port
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)

#show that there are no more null values
#sns.heatmap(dataset.isnull(),yticklabels=False)
#plt.show()

#need to convert non numerical values to numerical ones - embarked, sex and name
sex = pd.get_dummies(dataset['Sex'],drop_first = True)
embarked = pd.get_dummies(dataset['Embarked'])
#name more complicated - can't just use actual names, so extract title
titles = [n.split(', ')[1].split('.')[0] for n in dataset['Name']]
for i,t in enumerate(titles):
    if t == "Mlle" or t == "Ms":
        titles[i] = "Miss"
    elif t == "Mme":
        titles[i] = "Mrs"
    elif t != "Mr":
        titles[i] = "Rare"

titles = pd.get_dummies(titles)


dataset.drop(['Sex','Embarked','Name'],axis=1,inplace=True)
dataset = pd.concat([dataset,sex,embarked,titles],axis=1)

print(dataset.head(5))

#print(dataset.describe(include='all'))  #verify that all are now numeric

#now we have good numerical data

#visualize it
sns.heatmap(dataset.corr(), annot = True)

#plt.show()

#time to train the model

#split dataset into features (x) and output (y)
y = dataset['Survived']
x = dataset.drop(['Survived'],axis=1)

print(x.head(5))

#standardize inputs
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

from keras import Sequential
from keras.layers import Dense

classifier = Sequential()

#first hidden layer
classifier.add(Dense(8,activation='relu',kernel_initializer='random_normal',input_dim=13))

#second hidden layer
classifier.add(Dense(8, activation='relu',kernel_initializer='random_normal'))

#output layer
classifier.add(Dense(1,activation='sigmoid',kernel_initializer='random_normal'))

#compile the neural network
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fit the network
classifier.fit(x_train,y_train,batch_size=10,epochs=100)

eval_model = classifier.evaluate(x_train,y_train)
print(eval_model)

y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)   #take all predictions greater than 0.5 to be 1 (survived)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print(cm)

print("accuracy = " + str((cm[0][0] + cm[1][1])/len(y_test)))

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
