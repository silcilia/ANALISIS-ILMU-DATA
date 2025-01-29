import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
# Misalnya: data = pd.read_csv("data_yield.csv")
data = pd.DataFrame({
    'Farm_Area': np.random.uniform(1, 100, 100),
    'Irrigation_Type': np.random.choice(['Type1', 'Type2', 'Type3'], 100),
    'Soil_Type': np.random.choice(['Sandy', 'Loamy', 'Clay'], 100),
    'Crop_Type': np.random.choice(['CropA', 'CropB', 'CropC'], 100),
    'Season': np.random.choice(['Summer', 'Winter', 'Monsoon'], 100),
    'Water_Usage': np.random.uniform(50, 500, 100),
    'Fertilizer_Used': np.random.uniform(10, 200, 100),
    'Yield': np.random.uniform(10, 100, 100)
})

# Split data into features and target
X = data.drop(columns=['Yield'])
y = data['Yield']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline: OneHotEncoding for categorical variables, StandardScaler for numeric
categorical_features = ['Irrigation_Type', 'Soil_Type', 'Crop_Type', 'Season']
numeric_features = ['Farm_Area', 'Water_Usage', 'Fertilizer_Used']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Define a Random Forest model with pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Set up hyperparameter grid for tuning
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['auto', 'sqrt', 'log2']
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Retrieve the best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Predict on test set
y_pred = best_model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Display comparison of actual vs predicted values
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print(comparison.head(10))
