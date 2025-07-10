# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load the heart failure dataset
data = pd.read_csv('heart_failure.csv')
print(data.info())  # Display dataset information

# Display class distribution for the target variable
y = data['death_event']
print('Classes and number of values in the dataset', Counter(y))

# Select features and target variable
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]
# One-hot encode categorical variables if any
x = pd.get_dummies(x)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Standardize numeric features using ColumnTransformer
ct = ColumnTransformer([
    ('numeric', StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])
])
X_train = ct.fit_transform(X_train)
X_test = ct.fit_transform(X_test)

# Encode target labels as integers, then one-hot encode for neural network
le = LabelEncoder()
Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.transform(Y_test.astype(str))
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Build a simple neural network model
model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1],)))
model.add(Dense(12, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)

# Evaluate the model on the test set
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Loss', loss, 'Accuracy', acc)

# Make predictions and print classification report
y_estimate = model.predict(X_test, verbose=0)
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(Y_test, axis=1)
print(classification_report(y_true, y_estimate))