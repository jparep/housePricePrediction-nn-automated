# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from scikeras.wrappers import KerasRegressor

# Set random seed for Numpy and Tensorflow
myID = 373
np.random.seed(myID)
tf.random.set_seed(myID)

# Handling Outlier Class
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
    
    def fit(self, X, y=None):
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_bound = Q1 - self.factor * IQR
        self.upper_bound = Q3 + self.factor * IQR
        return self
    
    def transform(self, X, y=None):
        return X.apply(lambda x: x.clip(self.lower_bound[x.name], self.upper_bound[x.name]))

# House Price Predictor Class
class HousePricePredictor:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.model_before_hyperTune = None
        self.model_after_hyperTune = None
        self.best_params = None
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.dataPath)
        except FileNotFoundError:
            print(f'The file {self.dataPath} was not found.')
        except pd.errors.ParserError:
            print(f'Error parsing the file {self.dataPath}.')
        except Exception as e:
            print(f'An error occurred while loading the data: {e}')

    def define_features(self):
        if self.df is not None:
            self.num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()

    def numerical_features_transformer(self):
        return Pipeline(steps=[
            ('outliers', OutlierHandler()),
            ('imputer', IterativeImputer(max_iter=10, random_state=myID)),
            ('scaler', StandardScaler())
        ])
    
    def categorical_features_transformer(self):
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most-frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
    
    def preprocess_data(self):
        return ColumnTransformer(transformers=[
            ('num', self.numerical_features_transformer(), self.num_cols),
            ('cat', self.categorical_features_transformer(), self.cat_cols)                              
        ])
    
    def create_model(self, input_shape, layers=[128, 64]):
        model = Sequential()
        model.add(Dense(layers[0], input_shape=(input_shape,), activation=LeakyReLU(alpha=0.1)))
        for size in layers[1:]:
            model.add(Dense(size, activation=LeakyReLU(alpha=0.1)))
            model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        return model
    
    def train_model(self, X_train, y_train, input_shape):
        self.model_before_hyperTune = KerasRegressor(
            model=self.create_model,
            model__input_shape=input_shape,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        self.model_before_hyperTune.fit(X_train, y_train)
    
    def hyperparameter_tuning(self, X_train, y_train, input_shape):
        model = KerasRegressor(
            model=self.create_model,
            model__input_shape=input_shape,
            verbose=0
        )
        param_dist = {
            'model__layers': [[128, 64], [64, 32]],
            'batch_size': [32, 64],
            'epochs': [30, 60]
        }
        grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=12,
            cv=5,
            verbose=1,
            n_jobs=-1,
            random_state=myID
        )
        grid.fit(X_train, y_train)
        self.model_after_hyperTune = grid.best_estimator_
        self.best_params = grid.best_params_
        
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R Squared: {r2:.2f}")
    
    def run(self):
        self.load_data()
        if self.df is not None:
            # Define Features and target variables
            X, y = self.df.drop(['Id', 'SalePrice'], axis=1), self.df['SalePrice']
            self.define_features() # Split Numerical and Categorical variables
            preprocessor = self.preprocess_data() # Transform data
            
            # Split data into train and test datasets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=myID)
            # Fit and transform training dataset
            X_train_preprocessed = preprocessor.fit_transform(X_train)
            # Transform the test dataset with the same transformation
            X_test_preprocessed = preprocessor.transform(X_test)
            
            # Train the model without Hyperparameter Tuning
            self.train_model(X_train_preprocessed, y_train, X_train_preprocessed.shape[1])
            
            print(f"Model Evaluation Before Hyperparameter Tuning:")
            self.evaluate_model(self.model_before_hyperTune, X_test_preprocessed, y_test)
            
            # Hyperparameter tuning of the model
            self.hyperparameter_tuning(X_train_preprocessed, y_train, X_train_preprocessed.shape[1])
            
            print(f"Model Evaluation After Hyperparameter Tuning:")
            self.evaluate_model(self.model_after_hyperTune, X_test_preprocessed, y_test)

if __name__ == "__main__":
    predictor = HousePricePredictor("/content/drive/MyDrive/Projects/HousePricePrediction/data/house_data.csv")
    predictor.run()