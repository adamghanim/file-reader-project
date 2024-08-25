# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:55:30 2024

@author: adamg
"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

print("All libraries imported successfully.")
  # טעינת נתונים בעזרת
  
# טעינת נתונים בעזרת pandas
data = pd.read_csv('C:/Users/adamg/OneDrive/שולחן העבודה/pythone/homework/homework/בניה ומודעין עסקי/diabetes.csv')

# הצגת השורות הראשונות של הנתונים
print(data.head())

# משתני המטרה והמאפיינים
X = data.drop(columns=['Outcome'])  # משתנים עצמאיים (מאפיינים)
y = data['Outcome']  # משתנה תלוי (מטרה)

print("מאפיינים (X):")
print(X.head())
print("\nמשתנה מטרה (y):")
print(y.head())

from sklearn.model_selection import train_test_split

# חלוקת הנתונים לאימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("מספר דוגמאות בסט האימון:", X_train.shape[0])
print("מספר דוגמאות בסט הבדיקה:", X_test.shape[0])

from sklearn.tree import DecisionTreeClassifier

# יצירת מודל עץ החלטה
model = DecisionTreeClassifier(random_state=42)

# אימון המודל על סט האימון
model.fit(X_train, y_train)

print("המודל אומן בהצלחה.")

# תחזית על סט הבדיקה
y_pred = model.predict(X_test)

# חישוב דיוק המודל
accuracy = accuracy_score(y_test, y_pred)
print(f"דיוק המודל: {accuracy:.2f}")

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree



# הדמיית עץ ההחלטה
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], rounded=True)
plt.show()


