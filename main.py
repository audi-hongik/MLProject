import numpy as np
import os
import pandas as pd  # data processing, CSV file I/O
from sklearn.impute import SimpleImputer, KNNImputer  # imputation library
from impyute.imputation.cs import mice, fast_knn  # multiple imputation library
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce

data = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
target = data[['Global_Sales']]
# anxis =1 은 열을 삭제하겠다는 것을 뜻한다
# Replacing "tbd" values with np.nan and transforming column to float type
data["User_Score"] = data["User_Score"].replace("tbd", np.nan).astype(float)
data.drop(['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1, inplace=True)


# Analysis : NAN값 있는 행 제거
data_remove = data.dropna(axis=0)
remove_target = data_remove[['Global_Sales']]  # index 4
data_remove.drop(['Global_Sales'], axis=1, inplace=True)
# data_remove = data_remove.astype({'User_Score': 'float64'})

rm_data_test = list()
rm_data_train = list()
rm_target_test = list()
rm_target_train = list()

arr_data_remove = np.array(data_remove.values)
arr_remove_target = np.array(remove_target.values)
# training data 5460, test data 1365개 8:2 비율로 나눔
for i in range(arr_data_remove[:, :1].size):
    if i % 5 == 4:
        rm_data_test.append(arr_data_remove[i, :])
    else:
        rm_data_train.append(arr_data_remove[i, :])

for i in range(arr_remove_target.size):
    if i % 5 == 4:
        rm_target_test.append(arr_remove_target[i])
    else:
        rm_target_train.append(arr_remove_target[i])


# Imputation : 평균값(대표값)을 사용하여 데이터 보정
imp_mean = SimpleImputer(strategy='most_frequent')  # mean, median
imp_mean.fit(data)
arr_imp_mean = imp_mean.transform(data)
target_imp_mean = arr_imp_mean[:, 4:5]
np.delete(arr_imp_mean, 4, axis=1)

# data_remove = pd.DataFrame(arr_imp_mean)
# data_remove.info()
# print(data_remove)


imp_mean_test = list()
imp_mean_train = list()
imp_target_test = list()
imp_target_train = list()

# 16719 training data 13375, test data 3344 8:2 비율로 나눔
for i in range(arr_imp_mean[:, :1].size):
    if i % 5 == 4:
        imp_mean_test.append(arr_imp_mean[i, :])
    else:
        imp_mean_train.append(arr_imp_mean[i, :])

for i in range(target_imp_mean.size):
    if i % 5 == 4:
        imp_target_test.append(target_imp_mean[i])
    else:
        imp_target_train.append(target_imp_mean[i])

# TODO : null값 컬럼들 enum으로 변경 후 반올림
# print(data.isna().sum(axis=0))
# nan이 있는 컬럼 보정필요 - year(269), genre(2), publisher(54), rating(6769), developer(6623)
# 보정불필요 - critic_score, count(8582), user_score(6704), user_count(9129)
# s = data.dtypes == 'object'

year_arr = np.array(data[['Year_of_Release']].dropna(axis=0))
year_label = list()
genre_arr = np.array(data[['Genre']].dropna(axis=0))
encoder = ce.one_hot.OneHotEncoder()
categorical_subset = encoder.fit_transform(genre_arr)
# print(categorical_subset)

# for i in range(year_arr.size):
#     if year_arr[i]

# TODO : Multiple Imputation : Imputation -> Analysis -> Pooling 과정 거쳐 데이터 보정
# imputed_training = mice(data.to_numpy())
# print(imputed_training)

# TODO: KNN써서 data 보정
# imputer_knn = KNNImputer(n_neighbors=5)
# imputed_knn = imputer_knn.fit_transform(data)
# print(imputed_knn)

# imputed_knn2 = fast_knn(data, k=5);
# print(imputed_knn2)
# float형태의 data만 가능

# Name, NA_Sales, EU_Sales, JP_Sales, Other_Sales 제외
# Global_Sales 는 output

# print(data)
