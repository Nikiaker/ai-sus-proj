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


def basic_info():
    # podstawowe info o kolumnach
    data_w = pd.read_csv('data/AmesHousing.csv')
    data_w.columns = data_w.columns.str.replace(' ', '')
    data_w.drop(columns=['Order', 'PID'], inplace=True)
    data_w.info()
    return data_w


def class_info(data_w):
    # analiza klasy (kolumna SalePrice)
    (mu, sigma) = norm.fit(data_w['SalePrice'])
    min_price = data_w['SalePrice'].min()
    max_price = data_w['SalePrice'].max()

    plt.figure(figsize=(12, 6))
    sns.displot(data_w['SalePrice'], kde=True)
    plt.title('Rozkład wartości atrybutu SalePrice', fontsize=13)
    plt.xlabel("Cena domów w dolarach $", fontsize=12)
    plt.legend([
        '($\mu=$ {:.2f}, $\sigma=$ {:.2f}, min=${:.2f}, max=${:.2f})'.format(mu, sigma, min_price, max_price)
    ], loc='best')
    plt.show()


def correlation_info(data_w):
    # analiza korelacji kryteriów
    f, ax = plt.subplots(figsize=(30, 25))
    mat = data_w.select_dtypes(include=[np.number]).corr('pearson')
    mask = np.triu(np.ones_like(mat, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, annot=True,
                square=False, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

    # Print correlations with SalePrice
    print("Correlations with SalePrice:")
    print(mat['SalePrice'].sort_values(ascending=False))

def numeric_values_info(data_w: pd.DataFrame):
    numerical_data = data_w.select_dtypes("number").drop(columns=['SalePrice'])
    num_columns = numerical_data.shape[1]
    rows = int(np.ceil(num_columns / 4))  # Adjust rows dynamically
    numerical_data.hist(
        bins=20, figsize=(12, 22), edgecolor="black", layout=(rows, 4)
    )
    plt.subplots_adjust(hspace=0.8, wspace=0.8)
    plt.show()

def categorical_values_info(data_w: pd.DataFrame):
    string_data = data_w.select_dtypes(object)
    n_string_features = string_data.shape[1]
    nrows, ncols = ceil(n_string_features / 11), 11

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(14, 80))

    for feature_name, ax in zip_longest(string_data, axs.ravel()):
        if feature_name is None:
            # do not show the axis
            ax.axis("off")
            continue

        string_data[feature_name].value_counts().plot.barh(ax=ax)

    plt.subplots_adjust(hspace=0.2, wspace=0.8)
    plt.show()

def check_missing_values(data_w: pd.DataFrame):
    print(data_w.isnull().sum().sort_values(ascending=False).head(30))

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

    return ames_housing_preprocessed

def important_data_wiv(data_w: pd.DataFrame):
    #sns.boxplot(data=data_w, x = 'OverallQual', y='SalePrice')
    #plt.show()

    # top_20_values = data_w['GrLivArea'].value_counts().nlargest(100).index
    # filtered_data = data_w[data_w['GrLivArea'].isin(top_20_values)]
    # sns.boxplot(data=filtered_data, x='GrLivArea', y='SalePrice')
    # plt.xticks(rotation=45)
    # plt.show()

    top_20_values = data_w['GarageArea'].value_counts().nlargest(75).index
    filtered_data = data_w[data_w['GarageArea'].isin(top_20_values)]
    figure, ax = plt.subplots(1,2)
    sns.boxplot(data=data_w, x = 'GarageCars', y='SalePrice', ax = ax[0])
    sns.boxplot(data=filtered_data, x = 'GarageArea', y='SalePrice', ax = ax[1])
    plt.xticks(rotation=45)
    plt.show()

def main():
    data_w = basic_info()
    #class_info(data_w)
    #correlation_info(data_w)
    #numeric_values_info(data_w)
    #categorical_values_info(data_w)
    #check_missing_values(data_w)

    data_w_fixed = fix_missing_values(data_w)
    #check_missing_values(data_w_fixed)
    #numeric_values_info(data_w)
    #categorical_values_info(data_w)
    #correlation_info(data_w_fixed)
    important_data_wiv(data_w_fixed)

if __name__ == "__main__":
    main()