import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np
from itertools import zip_longest
from math import ceil
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score



def basic_info():
    # podstawowe info o kolumnach
    data_w = pd.read_csv('data/AmesHousing.csv')
    data_w.columns = data_w.columns.str.replace(' ', '')
    data_w.drop(columns=['Order', 'PID'], inplace=True)
    data_w.info()
    return data_w

def fix_missing_values(data_w: pd.DataFrame):
    data_w = data_w.copy().drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'LotFrontage'])
    target_name = "SalePrice"
    data, target = (
        data_w.drop(columns=target_name),
        data_w[target_name],
    )
    numerical_features = [
        "LotArea",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "TotRmsAbvGrd",
        "Fireplaces",
        "GarageCars",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
        target_name,
    ]
    categorical_features = data.columns.difference(numerical_features)

    most_frequent_imputer = SimpleImputer(strategy="most_frequent")
    mean_imputer = SimpleImputer(strategy="mean")

    preprocessor = make_column_transformer(
        (most_frequent_imputer, categorical_features),
        (mean_imputer, numerical_features),
    )
    ames_housing_preprocessed = pd.DataFrame(
        preprocessor.fit_transform(data_w),
        columns=categorical_features.tolist() + numerical_features,
    )
    ames_housing_preprocessed = ames_housing_preprocessed[data_w.columns]
    ames_housing_preprocessed = ames_housing_preprocessed.astype(
        data_w.dtypes
    )

    return pd.get_dummies(ames_housing_preprocessed)

def evaluate_models(data, target):
    # Definicja modeli
    models = {
        "Linear Regression": LinearRegression(),
        "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=100),
        "MLPRegressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

    # Definicja metryk
    scoring = {
        "MAE": make_scorer(mean_absolute_error),
        "MSE": make_scorer(mean_squared_error),
        #"R2": make_scorer(r2_score)
    }

    # 10-Fold Cross Validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    results = {}
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        model_results = {}
        for metric_name, metric in scoring.items():
            scores = cross_val_score(model, data, target, cv=kfold, scoring=metric)
            model_results[metric_name] = {
                "mean": scores.mean(),
                "std": scores.std()
            }
        results[model_name] = model_results

    # Wyświetlenie wyników
    for model_name, metrics in results.items():
        print(f"\nResults for {model_name}:")
        for metric_name, values in metrics.items():
            print(f"{metric_name}: Mean={values['mean']:.4f}, Std={values['std']:.4f}")

def main():
    target_name = "SalePrice"
    data_w = basic_info()
    data_w_fixed = fix_missing_values(data_w)
    data, target = (
        data_w_fixed.drop(columns=target_name),
        data_w_fixed[target_name],
    )

    evaluate_models(data, target)

if __name__ == "__main__":
    main()