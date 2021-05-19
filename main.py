
import numpy as np
import os
import pandas as pd  # data processing, CSV file I/O
from sklearn.impute import SimpleImputer, KNNImputer  # imputation library
from impyute.imputation.cs import mice, fast_knn  # multiple imputation library

data = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
# preview the first 5 lines of the loaded data
print(data)

# Analysis : NAN값 있는 행 제거
data_remove = data.dropna(axis=0)
data_remove = data_remove.astype({'User_Score': 'float64'})
# print(data_remove)
# TODO : input 나누기
# TODO : train, test 데이터 나누기

# Imputation : 평균값(대표값)을 사용하여 데이터 보정
imp_mean = SimpleImputer(strategy='most_frequent')  # mean, median
imp_mean.fit(data)
imputed_train_df = imp_mean.transform(data)
data_impute_mean = pd.DataFrame(imputed_train_df)
print(data_impute_mean)
# TODO : input 나누기
# TODO : train, test 데이터 나누기

# Multiple Imputation : Imputation -> Analysis -> Pooling 과정 거쳐 데이터 보정
# imputed_training = mice(data.to_numpy())
# print(imputed_training)

# KNN써서 data 보정
# imputer_knn = KNNImputer(n_neighbors=5)
# imputed_knn = imputer_knn.fit_transform(data)
# print(imputed_knn)

# imputed_knn2 = fast_knn(data, k=5);
# print(imputed_knn2)
# float형태의 data만 가능

# Name, NA_Sales, EU_Sales, JP_Sales, Other_Sales 제외
# Global_Sales 는 output

# anxis =1 은 열을 삭제하겠다는 것을 뜻한다
# data.drop(['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1, inplace=True)
# print(data)