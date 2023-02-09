# Dataset from Kaggle:https://www.kaggle.com/datasets/himanshunakrani/student-study-hours
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('score.csv')


#dataset splitting
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#visualizing the data to identify which regression will be applied
plt.scatter(x,y ,color= "blue")
plt.title("Score per Hours studying")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.3,random_state=0)


#lin.regession
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(x_train,y_train)

#visualizing the train data
plt.scatter(x_train,y_train ,color= "blue")
plt.plot(x_train,reg.predict(x_train), color= "red")
plt.title("Score per Hours studying")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#visualizing the test data
plt.scatter(x_test,y_test ,color= "blue")
plt.plot(x_train,reg.predict(x_train), color= "red")
plt.title("Score per Hours studying")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
