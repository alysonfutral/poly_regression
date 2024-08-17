
# Ridge Regression (Ridge): (Polynomial Regression)
# Think of regression as a way to predict an outcome (like price or sales) based on certain inputs (like temperature or advertising spend).
#Ridge regression is a specific type of regression that helps to prevent overfitting, which is when the model fits too closely to the training data but performs poorly on new, unseen data.
#It does this by adding a penalty to the model's coefficients (the numbers that determine the relationship between inputs and outcomes), discouraging them from becoming too large.
#This is helpful when there are many input variables, and some of them might be correlated with each other.



# ridge regression helps to prevent overfitting (when the model fits too closely to the training data but performs poorly on new, unseen data)
from sklearn.linear_model import Ridge



# Polynomial Features (PolynomialFeatures):
# Sometimes, the relationship between inputs and outcomes isn't straight or simple; it might be curved or nonlinear.
# Polynomial features help capture these more complex relationships by creating new input features based on the existing ones.
# For example, if you have inputs like "temperature" and "humidity," polynomial features might create new features like "temperature squared" or "temperature times humidity."
# This can help the model better understand the data and make more accurate predictions.




# Polynomial features help capture these more complex relationships by creating new input features based on the existing ones
from sklearn.preprocessing import PolynomialFeatures


# Pipeline (make_pipeline):
# Imagine you have several steps to do before making a prediction, like cleaning the data, transforming it, and then fitting a model.
# A pipeline is like an assembly line that connects these steps together, ensuring that the output of one step becomes the input for the next.
# It's handy because it keeps your code organized and reduces the chance of mistakes.
# make_pipeline is a shortcut for creating these pipelines in scikit-learn without having to write out all the steps individually.



# A pipeline is like an assembly line that connects these steps (PRIOR TO PREDICTIONS) together, ensuring that the output of one step becomes the input for the next.
from sklearn.pipeline import make_pipeline


# time of day
input_data = [
  [9],
  [10],
  [11],
  [12],
  [13],
  [14],
  [15],
  [16],
]
# out put waiting time
output_data = [0,10,20,30,40,30,20,10]

#In the first line of code, we create our polynomial regression model based on sklearn's PolynomialFeatures, but we use what is known as a pipeline to help handle some of the more difficult tasks that come with polynomial regression. Unlike a linear model, we cannot simply fit the input and output into the polynomial model. The pipeline takes in the transformed input data and fits it as polynomial input for the input and output data of the model.


#The next difference to our code is that we provided a degree for the polynomial features. Just as we showed with our discussion on how to calculate polynomial regression mathematically, we need to provide a degree for our formula; with most parabolas like this one it is just a degree of 2. This basically works out to the number of changes in direction for our line.
#Lastly, we added the Ridge() parameter, which lets the program know how we are going to handle our polynomial regression.
#This will create the polynomial regressive model according to the transformed data from the PolynomialFeatures.

# model for poly regression with a degree of 2, works out to the number of changes in direction for our line.
model = make_pipeline(PolynomialFeatures(2), Ridge())

#– It calculates the parameters or weights on the training data (e.g. parameters returned by coef() in case of Linear Regression) and saves them as an internal object state.
model.fit(input_data, output_data)

#predict the data
print("At 11:30, our model says that we will have to wait: ")
print(model.predict([[11.5]]))
print("minutes.")

print()

############################################################################
# CUSTOM INPUT DATA

# day of week, time of day, customers that day
input_data = [
  [1, 9, 23],
  [1, 10, 54],
  [1, 11, 84],
  [1, 12, 25],
  [1, 13, 47],
  [1, 14, 36],
  [1, 15, 55],
  [1, 16, 7],
  [2, 9, 18],
  [2, 10, 45],
  [2, 11, 88],
  [2, 12, 6],
  [2, 13, 23],
  [2, 14, 36],
  [2, 15, 48],
  [2, 16, 33],

# estimation of waiting time
output_data = [0,10,20,30,40,30,20,10,0,20,40,60,80,60,40,20]

model = make_pipeline(PolynomialFeatures(2), Ridge())
model.fit(input_data, output_data)

print("At 15:00, our model says that we will have to wait: ")
print(model.predict([[1, 15, 30]]))
print("minutes.")
