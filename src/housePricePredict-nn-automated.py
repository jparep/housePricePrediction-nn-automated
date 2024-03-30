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
    