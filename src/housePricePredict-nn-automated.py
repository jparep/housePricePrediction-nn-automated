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
from sklearn.metrics import mean_square_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, LeakyReLU
from scikeras.wrappers import KerasRegressor

# Set random seed fro Numpy and Tensorflow
myID = 373
np.random.seed(myID)
tf.random.set_seed(myID)

# Handing Outlier Class
class OutlierHandlerClass(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5) -> None:
        self.factor = factor
    
    def fit(self, X, y=None):
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_bound = Q1 - self.factor * IQR
        self.upper_bound = Q3 + self.factor * IQR
        return self
    
    def transform(self, X, y=None):
        return X[(X >= self.lower_bound) & (X <= self.upper_bound)]
    
# House Price Predictor Class
class HousePricePredictorClass:
    def __init__(self, dataPath) -> None:
        self.dataPath = dataPath
        self.model_before_hyperTune = None
        self.model_after_hyperTune = None
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        try:
            self.df = pd.read_csv(self.dataPath)
        except FileNotFoundError:
            print(f'The file {self.dataPath} was not found.')
        except pd.errors.ParserError:
            print(f'Error parsing the file {self.dataPath}.')
        except Exception as e:
            print(f'An error occured while loading the data: {e}')

    def define_features(self) -> type:
        if self.df is not None:
            self.num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
    
    def numerical_features_transformer(self) -> Pipeline:
        return Pipeline(steps=[
            ('outliers', OutlierHandlerClass()),
            ('imputer', IterativeImputer(max_iter=10, random_state=myID)),
            ('scaler', StandardScaler())
        ])
    
    def categorical_features_transformer(self) -> Pipeline:
        return Pipeline(self[
            ('imputer', SimpleImputer(strategy='most-frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
    
    def preprocess_data(self) -> ColumnTransformer:
        return ColumnTransformer(transformers=[
            ('num', self.numerical_features_transformer(), self.num_cols),
            ('cat', self.categorical_features_transformer(), self.cat_cols)                              
        ])
    
    def create_model(self, inpute_shape, layers=[128, 64]) -> Sequential:
        model = Sequential([
            Dense(layers[0], inpute_shape=(inpute_shape,), activation=LeakyReLU),
            *[layer for size in layers[1:] for layer in (Dense(size, activation=LeakyReLU, Dropout=0.2))],
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', matrics=['mean_squared_error'])
        return model
    
    def train_model(self, X_train, y_train, input_shape) -> KerasRegressor:
        self.model_before_hyperTune = KerasRegressor(
            model = self.create_model,
            model__inpute_shape = input_shape,
            epochs = 10,
            batch_size = 32,
            verbose = 0
        )
        
        self.model_before_hyperTune.fit(X_train, y_train)
    
    def hyperparameter_tuning(self, X_train, y_train, inpute_shape) -> RandomizedSearchCV:
        model = KerasRegressor(
            model  = self.create_model,
            model__inpute_shape = inpute_shape,
            verbose=0
        )
        
        param_dist = {
            'model__layers': [[128, 64], [64, 32]],
            'batch_size': [32, 64],
            'epochs': [30, 60]
        }
        
        