# house_price_predictor.py
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from scikeras.wrappers import KerasRegressor

from configuration import Config

logging.basicConfig(level=logging.INFO)

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
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].clip(lower=self.lower_bound[col], upper=self.upper_bound[col])
        return X

class HousePricePredictor:
    def __init__(self, config):
        self.config = config
        np.random.seed(config.random_seed)

    def load_data(self):
        logging.info(f"Loading data from {self.config.data_path}")
        try:
            df = pd.read_csv(self.config.data_path)
            df.drop('Id', axis=1, inplace=True, errors='ignore')
            y = df.pop('SalePrice') if 'SalePrice' in df.columns else None
            return df, y
        except Exception as e:
            logging.error(f'Error loading the data: {e}')
            return None, None

    def _define_features(self, X):
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    def _numerical_pipeline(self):
        return Pipeline(steps=[
            ('outliers', OutlierHandler()),
            ('imputer', IterativeImputer(max_iter=10, random_state=self.config.random_seed)),
            ('scaler', StandardScaler())
        ])

    def _categorical_pipeline(self):
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

    def _build_preprocessor(self):
        return ColumnTransformer(
            transformers=[
                ('num', self._numerical_pipeline(), self.num_cols),
                ('cat', self._categorical_pipeline(), self.cat_cols)
            ]
        )

    def create_model(self, input_shape):
        model = Sequential([
            Dense(128, input_shape=(input_shape,), activation=LeakyReLU(alpha=0.1)),
            Dense(64, activation=LeakyReLU(alpha=0.1)),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        return model

    def train_model(self, X, y):
        self._define_features(X)
        preprocessor = self._build_preprocessor()
        X_preprocessed = preprocessor.fit_transform(X)
        input_shape = X_preprocessed.shape[1]
        model = KerasRegressor(model=self.create_model, model__input_shape=input_shape, epochs=100, batch_size=32)

        param_dist = {
            'model__epochs': [50, 100],
            'model__batch_size': [32, 64],
            'model__model__layers': [[128, 64], [64, 32]]
        }

        grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=self.config.n_iter_search,
            cv=self.config.cv_folds,
            random_state=self.config.random_seed
        )

        grid.fit(X_preprocessed, y)
        self.model = grid.best_estimator_
        logging.info(f"Best parameters: {grid.best_params_}")

    def evaluate_model(self, X, y):
        preprocessor = self._build_preprocessor()
        X_preprocessed = preprocessor.transform(X)

        y_pred = self.model.predict(X_preprocessed)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        logging.info(f"Mean Squared Error: {mse:.2f}")
        logging.info(f"R Squared: {r2:.2f}")

    def run(self):
        X, y = self.load_data()
        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size, random_state=self.config.random_seed)
            self.train_model(X_train, y_train)
            self.evaluate_model(X_test, y_test)

if __name__ == "__main__":
    config = Config()
    predictor = HousePricePredictor(config)
    predictor.run()
