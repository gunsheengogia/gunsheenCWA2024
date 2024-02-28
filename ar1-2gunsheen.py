# This program takes three values from a CSV file and compares them to predict a fourth value
# This is given the fancy name "Multiple Linear Regression".
# It's like a bunch of linear trendlines mashed up together to allow a few more extra variables.
# The model which given ABC to predict X then works like this:
# predicted_X_Value = predict_mood(A,B,C)
# print("The predicted value is", predicted_X_Value)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv('AR1-3.csv')

X = data[['temprature_min', 'temprature_max', 'temprature_mean']]
Y = data['Mood_Score']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print("Multiple Linear Regression Model Complete!")
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")
"""
def interpret_mse(mse):
    if mse < 10:
        return "Excellent model accuracy. This is the SHIZ."
    elif mse < 20:
        return "Good model accuracy. A fine auld model."
    elif mse < 30:
        return "Average model accuracy. That'll do pig."
    elif mse < 40:
        return "Below average model accuracy. Don't bet your house on this being true."
    else:
        return "Poor model accuracy! Get better data or try another fit like polyfit. This shirt ain't linear. \n"
mse_remark = interpret_mse(mse)
print("How good is this model? ", mse_remark)
def predict_mood(hours_of_sunlight, sunlight_intensity, peak_sunlight_intensity):
    df = pd.DataFrame([[hours_of_sunlight, sunlight_intensity, peak_sunlight_intensity]],
                      columns=['Hours_light', 'Intensity_Light', 'Peak_Light'])
    return model.predict(df)[0]
print("")
print("USER CHOOSES 3 LIGHT LEVELS MODE")
hours = int(input("Enter sunlight hours. Can be any integer from 0-24"))
sun = float(input("Enter average sunlight intensity. Can be anything from 1-800 "))
peak = float(input("Enter peak sunlight level. Can be anything from 1-800"))

predicted_mood = predict_mood(hours, sun, peak)  # Example values
print("\n The Predicted Mood Score for the values entered is", predicted_mood)
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 1")
print("Let's test what the mood will be if the sunlight is very low")
sunlight_hours = 2
average_sunlight = 100
peak = 200
mood_if_littleSun = predict_mood(sunlight_hours, average_sunlight, peak)  
print("\n The low sun score mood is", mood_if_littleSun)
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 2")
print("Let's test what the mood will be if the sunlight is very high")
sunlight_hours = 15
average_sunlight = 600
peak = 800
mood_if_LoadsaSun = predict_mood(sunlight_hours, average_sunlight, peak)  
print("\n The higher sun score mood is", mood_if_LoadsaSun)
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 3")
print("Let's test if average sunlight is more important than peak sunlight")
print("We will keep the hours (A) the same and double the others (B) and (C) one at a time")
print("")
print("Let's get a baseline from fairly average values...")
sunlight_hours = 6
average_sunlight = 300
peak = 300
baseline_mood = predict_mood(sunlight_hours, average_sunlight, peak)  
print("\n The baseline Score mood is", baseline_mood)
print("")
print("Let's double the average sunlight...")
sunlight_hours = 6
average_sunlight = 600
peak = 300
doubleAverageOutcome = predict_mood(sunlight_hours, average_sunlight, peak)  # Example values
print("The double sunlight mood is", doubleAverageOutcome)
print("")
print("Let's double the peak sunlight...")
sunlight_hours = 6
average_sunlight = 300
peak = 600
doublePeakOutcome = predict_mood(sunlight_hours, average_sunlight, peak)  # Example values
print("The double peak mood is", doublePeakOutcome)
print("")
print("OUTCOME:")
if doubleAverageOutcome > doublePeakOutcome:
    print("It's the average that improves mood the most")
else:
    print("It's the peak that improves mood the most")
import matplotlib.pyplot as plt
variable_names = ['Mood if Little Sun', 'Mood if Loadsa Sun',]
values = [mood_if_littleSun, mood_if_LoadsaSun]
plt.bar(variable_names, values)
plt.xlabel('Ammount of Sun')
plt.ylabel('Moodiness')
plt.title('Bar Chart of WHAT-IF Q1, Q2 Outcomes')
plt.show()
import matplotlib.pyplot as plt
variable_names = ['Mood if Double Average', 'Mood if Double Peak',]
values = [doubleAverageOutcome, doublePeakOutcome]
plt.bar(variable_names, values)
plt.xlabel('Ammount of Sun')
plt.ylabel('Moodiness')
plt.title('Bar Chart of WHAT-IF Q3 Outcome')
plt.show()
"""