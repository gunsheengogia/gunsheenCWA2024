# This program takes three values from a CSV file and compares them to predict a fourth value
# This program then attempts to execute a "Multiple Linear Regression" AR1
# using 3 independent variables and one dependent variable.       AR1
# The program uses input parameters from the user to make a prediction   AR1
# The program the considers a number of WHAT-IF scenarios using the "trained" model AR2
# to make further predictions
# The program then produces the outcomes from above in a graphical format   AR3
# My program is processing a dataset that originates from an embedded system which senses light
# and uses that data along with user input to test/train a model which predicts mood
# Some Standards: All functions will be at the top of the code, All import statements will be at the top of the code

#Import Statements
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Prediction function to predict using 3 independent variables and one dependent variable. 
def predict_mood(temprature_min, temprature_max, temprature_mean):
    df = pd.DataFrame([[temprature_min, temprature_max, temprature_mean]],
                      columns=['temprature_min', 'temprature_max', 'temprature_mean'])
    return my_model.predict(df)[0]
# Training the model
# first Load your dataset
data = pd.read_csv('AR1-3.csv') # This dataset is a copy of the output from BR1-3, copied so as to create a backup of BR1-3 data

# Define your independent variables (features) and dependent variable (target)
X = data[['temprature_min', 'temprature_max', 'temprature_mean']]
Y = data['avg_mood']

# Splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Creating the Linear Regression model
my_model = LinearRegression()

# Fitting the model with the training data
my_model.fit(X_train, Y_train)

# Predicting mood scores for the test set
Y_pred = my_model.predict(X_test)


print("My Multiple Linear Regression Model is now Complete!")

# Making a prediction using the model
# Let the user enter their own 3 parameters
# Note 2 different datatypes
print("")
print("USER CHOOSES 3 TEMPERATURE LEVELS MODES")
temperature_now = int(input("Enter the temperature of your room now. Can be any integer from 0-30 degrees celcius "))
average_temperature = float(input("Enter average temperature level of your room. Can be anything from 0-40 degrees celcius "))
peak = float(input("Enter peak temperature level of your room. Can be anything from 0-40 degrees celcius "))


predicted_mood = predict_mood(temperature_now, average_temperature, peak)  
print("\n The Predicted Mood Score for the values entered is", predicted_mood)

#___________________What If Questions AR2 _________
# WHAT-IF Q1
# What is will your mood be with low values given to the 3 params?
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 1")
print("Let's test what the mood will be if the temperature is very low")
temperature_now = 10
#average_temperature = 8
#peak = 20
mood_if_temperature_now = predict_mood(temperature_now, average_temperature, peak)  
print("\n The low temperature score mood is", mood_if_temperature_now)

# WHAT-IF Q2
# What is will your mood be with high values given to the 3 params?
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 2")
print("Let's test what the mood will be if the temperature is very high")
#temperature_now = 30
average_temperature = 20
#peak = 40
mood_if_average_temperature = predict_mood(temperature_now, average_temperature, peak)  
print("\n The higher temperature score mood is", mood_if_average_temperature)

# WHAT IF Q3
# # What is will your mood be with middle/normal values given to the 3 params?
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 3")
print("Let's test if average temperature is more important than peak temperature")
print("")
print("Let's get a baseline from fairly average values")
#temperature_now = 20
#average_temperature = 18
peak = 30
baseline_mood = predict_mood(temperature_now, average_temperature, peak)  
print("\n The baseline Score mood is", baseline_mood)
print("")

# Data: names of the variables and their values for the Bar Chart    AR3
# AR3 Users can view data in a graphical format which displays information such as their progress
#using the system or the results of a ‘what if’ scenario.

# Data: names of the variables and their values for the chart
variable_names = ['temperature now', 'average temperature','temperature peak']
values = [mood_if_temperature_now, mood_if_average_temperature, peak]

# Creating the bar chart
plt.bar(variable_names, values)

# Adding labels and title
plt.xlabel('Amount of temperature')
plt.ylabel('Mood')
plt.title('Bar Chart of all 3 WHAT-IFs Predictions')

# Show the plot
plt.show()