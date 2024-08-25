# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:35:53 2024

@author: adamg
"""
import pandas as pd

# קריאה ראשונית של הקובץ ללא הפרדה
data = pd.read_csv('C:/Users/adamg/OneDrive/שולחן העבודה/bank.csv', header=None)

# הפרדת העמודות על ידי שימוש בפונקציה split
data = data[0].str.split(';', expand=True)

# שימוש בשורה הראשונה ככותרות העמודות
data.columns = data.iloc[0]  # שורה ראשונה ככותרות
data = data[1:]  # הסרת השורה הראשונה מהנתונים

# הצגת חמש השורות הראשונות לאחר הוספת הכותרות והסרת השורה המיותרת
print(data.head())

# שמירת הקובץ לאחר ההפרדה (אופציונלי)
data.to_csv('C:/Users/adamg/OneDrive/שולחן העבודה/bank_fixed.csv', index=False)

