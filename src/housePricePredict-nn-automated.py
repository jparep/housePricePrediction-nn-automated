# Import Libraries
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from scikeras.wrappers import KerasRegressor


from configuration import config

logging.baseConfig(level=logging.INFO)

# Handle Outliers class
class OutlierHander(BaseEstimator, TransformerMixin):
    def __init__(self, factor =1.5) -> None:
        self.factor = factor
    
    def fir(self, X, y=None):
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_bound = Q1 - self.factor * IQR
        self.upper_bound = Q3  + self.factor * IQR
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].clip(lower=self.lower_bound[col], upper=self.upper_bound[col])
        
        return X


# House price predictor class
class HousePricePredictor:
    def __init__(self, config) -> None:
        self.config = config
        np.random.seed(config.random_seeed)
    
    def load_data(self):
        logging.info(f"Loading data from {self.config.data_path}")
        try:
            df =pd.read_csv(self.config.data_path)
            y = df.pop('SalePrice') if 'SalePrice' in df.columns else None
            X = df.drop(['Id', 'SalePrice'], axis=1, inplace=True, errors='ignore')
            return X, y
        except Exception as e:
            logging.error(f"Error loading the data: {e}")
            return None, None