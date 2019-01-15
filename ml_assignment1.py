import pandas as pd  #importing pandas library
import matplotlib.pyplot as plt   #importing matplotlib.pyplot

#loading the dataset into boston object
from sklearn.datasets import load_boston
boston = load_boston()

#loading the data into bos dataframe
bos = pd.DataFrame(boston.data)

#renaming the columns with it's feature names
bos.columns = boston.feature_names

#adding the price column to the bos
bos['PRICE'] = boston.target

#printing the top 5 records
print(bos.head())

#differentiating independent features and storing them in X 
X = bos.iloc[:, :-1].values

#differentiating dependent(output) features from dataframe and storing them in Y
y = bos.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#calculating the accuracy of model using r2_score 
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)

#plotting the expected value v/s predicted value
plt.scatter(y_test,y_pred,color='blue')
plt.title('expected value v/s predicted value')
plt.xlabel('expected value')
plt.ylabel('predicted value')
plt.show()
