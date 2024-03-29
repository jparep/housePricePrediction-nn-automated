# House Price Predictor Automation

## Overview
The House Price Predictor is a machine learning application designed to predict house prices based on various features using a neural network model. It automates the entire machine learning workflow including data preprocessing, model training, hyperparameter tuning, and evaluation.

## Installation

### Prerequisites
- Python 3.8+
- NumPy
- pandas
- scikit-learn
- TensorFlow
- Keras
- SciKeras

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/housePricePredictor-nn-automation.git
   cd house-price-predictor
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
To run the House Price Predictor, execute the following command in the terminal:
```sh
python housePricePrediction-nn-automation.py
```

Make sure to replace `"/content/drive/MyDrive/Projects/HousePricePrediction/data/house_data.csv"` in the `HousePricePredictor` class instantiation with the path to your CSV dataset.

## Features
- **Data Preprocessing**: Automated handling of numerical and categorical data.
- **Model Training**: Neural network model implemented in TensorFlow/Keras.
- **Hyperparameter Tuning**: Optimized model parameters using RandomizedSearchCV.
- **Model Evaluation**: Computed metrics such as MSE and R2 to evaluate model performance.

## Code Structure
- `load_data`: Function to load the dataset and handle exceptions.
- `numerical_transformer` and `categorical_transformer`: Functions to create preprocessing pipelines for numerical and categorical features.
- `preprocess_data`: Function that applies the preprocessing pipelines to the dataset.
- `create_model`: Function to define the neural network model.
- `train_model`: Function to train the model on the dataset.
- `hyperparameter_tuning`: Function to perform hyperparameter tuning on the model.
- `evaluate_model`: Function to evaluate the model's performance.
- `HousePricePredictor`: Class that orchestrates the model training and evaluation process.

## Data Format
The dataset should be a CSV file with columns corresponding to house features and a `SalePrice` column for the target variable. Adjust the script to fit your specific dataset structure and preprocessing needs.

## Customization
- The neural network architecture and hyperparameters can be adjusted in the `create_model` and `hyperparameter_tuning` functions.
- Data preprocessing steps can be customized in the `numerical_transformer` and `categorical_transformer` functions.

## License
Specify the license under which the project is released, such as MIT, GPL, Apache 2.0, etc.

## Contributors
- [Your Name](https://github.com/jparep)

## Acknowledgments
List any contributors, data sources, or organizations that have supported this project.
