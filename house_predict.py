import pandas as pd
from sklearn import linear_model
import pickle
import joblib

# Loading 
train_df = pd.read_csv('train.csv')

# Creating the model object
model = linear_model.LinearRegression()

# Fitting the model with X_train - area, y_train - price
model.fit(train_df[['area']], train_df['price'])

# Checking coefficient - m
print("coefficient:", model.coef_)

# Checking intercept - b
print("intercept", model.intercept_)

# Predicting model values - area = 5000
prediction = model.predict([[5000]])
print("Predicted price for an area of 5000 sq. units:", prediction[0])

# Saving and loading using pickle
# Save the model to a file named 'model_pkl'
with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)

# Load the saved model from 'model_pkl'
with open('model_pkl', 'rb') as f:
    lr = pickle.load(f)

# Check prediction using the loaded model
lr_prediction = lr.predict([[5000]])
print("Prediction using pickle-loaded model:", lr_prediction[0])

# Saving and loading using joblib
# Save the model to a file named 'model_jlib'
joblib.dump(model, 'model_jlib')

# Load the saved model from 'model_jlib'
try:
    m_jlib = joblib.load('model_jlib')
    m_jlib_prediction = m_jlib.predict([[5000]])
    print("Prediction using joblib-loaded model:", m_jlib_prediction[0])
except Exception as e:
    print("Error loading or using joblib model:", e)

