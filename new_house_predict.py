import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load the train dataset
train_data = pd.DataFrame({
    'type': ['condo', 'landed', 'landed', 'landed', 'landed'],
    'postcode': [40000, 40500, 40500, 40000, 40000],
    'area': [2600, 3000, 3200, 3600, 4000],
    'room': [3, 4, 5, 4, 4],
    'price': [550000, 565000, 610000, 680000, 725000]
})

# Prepare features (X_train) and target variable (y_train)
X_train = train_data[['postcode', 'area', 'room']]
y_train = train_data['price']

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Prepare new input data for prediction
new_input = pd.DataFrame({
    'type': ['landed'],
    'postcode': [40000],
    'area': [2600],
    'room': [3]
})

# Load the saved model using pickle
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make predictions for the new input
new_prediction = loaded_model.predict(new_input[['postcode', 'area', 'room']])
print("Predicted price for the new input:", new_prediction[0])
