# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 20:32:48 2024

@author: adamg
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# קריאה של הקובץ המתוקן
data = pd.read_csv('C:/Users/adamg/OneDrive/שולחן העבודה/pythone/homework/homework/בניה ומודעין עסקי/bank_fixed.csv')

# המרת משתנים קטגוריאליים לערכים מספריים
data['default'] = data['default'].map({'no': 0, 'yes': 1})
data['housing'] = data['housing'].map({'no': 0, 'yes': 1})
data['loan'] = data['loan'].map({'no': 0, 'yes': 1})
data['y'] = data['y'].map({'no': 0, 'yes': 1})

# המרת כל המשתנים הקטגוריאליים הנותרים באמצעות One-Hot Encoding
data = pd.get_dummies(data, drop_first=True)

# חלוקת הנתונים לסטי אימון ובדיקה
features = data.drop(columns='y')
target = data['y']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# בניית המודל
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# הערכת המודל
y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# הצגת דוח דירוג המודל
print(metrics.classification_report(y_test, y_pred))

# חשיבות התכונות
feature_importances = pd.Series(model.feature_importances_, index=features.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# הצגת החשיבות בצורה גרפית עם שיפורים
plt.figure(figsize=(12,8))  # הגדלת גודל הגרף
sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis')  # שימוש בפלטת צבעים
plt.xlabel('Feature Importance Score', fontsize=14)  # גודל גופן גדול יותר
plt.ylabel('Features', fontsize=14)  # גודל גופן גדול יותר
plt.title("Visualizing Important Features", fontsize=16)  # גודל גופן גדול יותר
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
