import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('student_data.csv')

# Encode Previous Grade
grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
df['Previous Grade'] = df['Previous Grade'].map(grade_map)

# Encode target labels
le = LabelEncoder()
df['Performance'] = le.fit_transform(df['Performance'])

# Features and target
X = df[['Study Hours', 'Attendance', 'Previous Grade']]
y = df['Performance']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model_performance.pkl', 'wb'))