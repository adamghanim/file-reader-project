# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:15:25 2024

@author: adamg
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported successfully.")

# טעינת נתונים
data = pd.read_csv('C:/Users/adamg/OneDrive/שולחן העבודה/pythone/homework/homework/בניה ומודעין עסקי/diabetes.csv')

# הצגת השורות הראשונות של הנתונים
print(data.head())

# בחירת משתנה המטרה והמשתנים העצמאיים
X = data.drop(columns=['Outcome'])  # משתנים עצמאיים
y = data['Outcome']  # משתנה מטרה

print("מאפיינים (X):")
print(X.head())
print("\nמשתנה מטרה (y):")
print(y.head())

from sklearn.model_selection import train_test_split

# חלוקת הנתונים לאימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("מספר דוגמאות בסט האימון:", X_train.shape[0])
print("מספר דוגמאות בסט הבדיקה:", X_test.shape[0])


from sklearn.naive_bayes import GaussianNB

# יצירת מודל Naive Bayes
model = GaussianNB()

# אימון המודל על סט האימון
model.fit(X_train, y_train)

print("המודל אומן בהצלחה.")


# תחזית על סט הבדיקה
y_pred = model.predict(X_test)

# חישוב דיוק המודל
accuracy = accuracy_score(y_test, y_pred)
print(f"דיוק המודל: {accuracy:.2f}")

# יצירת מטריצת בלבול
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nמטריצת בלבול:")
print(conf_matrix)

# יצירת דוח סיווג
class_report = classification_report(y_test, y_pred)
print("\nדוח סיווג:")
print(class_report)
