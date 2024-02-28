#import statements here
import pandas as pd
from statistics import mean
import csv
import serial
from time import sleep
#function to give a remark on my mood based on avg_mood value
def interpret_mood(avg_mood):
    if avg_mood >= 9:
        return "My mood is grand today!!"
    elif avg_mood <  5:
        return "A little meh today"
    elif 5 <= avg_mood < 9:
        return "Getting there"
    else:
        return "Not sure for today \n"
#Take them in as integers, as all inputs default to strings
physical_wellness = int(input("On a scale of 1-10 from healthy to unsatisfactory, how is your physical wellbeing? "))
sleep_wellness = int(input("On a scale of 1-10 from amazing to trashy, how well do you sleep at night? "))
mental_wellness = int(input("On a scale of 1-10 from good to bad, how is your mental wellbeing? "))
avg_mood = round(mean([physical_wellness,sleep_wellness,mental_wellness]),2)
mood_remark = interpret_mood(avg_mood)
print("My Average mood today is ",mood_remark, " ", avg_mood)
df = pd.read_csv('gunsheen105639.csv')
print(df)
#Convert 'Timestamp' column to datetime, is it necessary
#df['time (seconds)'] = pd.to_datetime(df['time (seconds)'], errors='coerce')
temperature_min = df['temperature'].min()
temperature_max = df['temperature'].max()
temperature_mean = df['temperature'].mean()
print (temperature_max,temperature_mean, avg_mood)
f = open("gunsheenBR1-3_2.csv", "a", newline='')
csver = csv.writer(f)
#csver.writerow(["temprature_min","temprature_max","temprature_mean","avg_mood"])
csver.writerow([temperature_min, temperature_max, temperature_mean, avg_mood])
f.close()
#Data validation BR2
if not isinstance(temperature_min, float):
    temperature_min = float(temperature_min)
if not isinstance(temperature_min, float):
    temperature_min = float(temperature_min)
if not isinstance(temperature_max, float):
    temperature_max = float(temperature_max)
if not isinstance(temperature_mean, float):
    temperature_mean = float(temperature_mean)