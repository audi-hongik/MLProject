import numpy as np
import os
import pandas as pd  # data processing, CSV file I/O
from sklearn.impute import SimpleImputer, KNNImputer  # imputation library
from impyute.imputation.cs import mice  # multiple imputation library
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



data = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
target = data[['Global_Sales']]
# anxis =1 은 열을 삭제하겠다는 것을 뜻한다
# Replacing "tbd" values
data = data.replace("tbd", np.NaN)
data.drop(['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Developer'], axis=1, inplace=True)
data = data.astype({'User_Score': 'float64'})

# float형태의 data만 가능 => one hot encoding
data_subset = data[["Year_of_Release", "Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]]
encoding_subset = data[["Platform", "Genre", "Publisher", "Rating"]]
encoder = ce.one_hot.OneHotEncoder()
encoding_subset = encoder.fit_transform(encoding_subset)
encoding_subset = encoding_subset.reset_index()
data = pd.concat([data_subset, encoding_subset], axis=1)

# Analysis : NAN값 있는 행 제거
data_remove = data.dropna(axis=0)
remove_target = data_remove[['Global_Sales']]  # index 4
data_remove.drop(['Global_Sales'], axis=1, inplace=True)

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


# Multiple Imputation : Imputation -> Analysis -> Pooling 과정 거쳐 데이터 보정
imp_mice = mice(data_subset.to_numpy())
imp_mice = pd.DataFrame(imp_mice)
mice_features = pd.concat([imp_mice, encoding_subset], axis=1)

mice_data = mice_features.drop(1, axis=1).values
mice_target = mice_features[1].values

mice_test = list()
mice_train = list()
mice_target_test = list()
mice_target_train = list()

16719 training data 13375, test data 3344 8:2 비율로 나눔
for i in range(mice_data[:, :1].size):
    if i % 5 == 4:
        mice_test.append(mice_data[i, :])
    else:
        mice_train.append(mice_data[i, :])

for i in range(mice_target.size):
    if i % 5 == 4:
        mice_target_test.append(mice_target[i])
    else:
        mice_target_train.append(mice_target[i])

# KNN써서 data 보정
# nan이 있는 컬럼 보정필요 - year(269), genre(2), publisher(54), rating(6769), developer(6623)
# 보정불필요 - critic_score, count(8582), user_score(6704), user_count(9129)

imputer_knn = KNNImputer(n_neighbors=5)
imputed_knn = imputer_knn.fit_transform(data_subset.to_numpy())
imputed_knn = pd.DataFrame(imputed_knn)
knn_features = pd.concat([imputed_knn, encoding_subset], axis=1)

knn_data = knn_features.drop(1, axis=1).values
knn_target = knn_features[1].values

imp_knn_test = list()
imp_knn_train = list()
imp_knn_target_test = list()
imp_knn_target_train = list()

# 16719 training data 13375, test data 3344 8:2 비율로 나눔
for i in range(knn_data[:, :1].size):
    if i % 5 == 4:
        imp_knn_test.append(knn_data[i, :])
    else:
        imp_knn_train.append(knn_data[i, :])

for i in range(knn_target.size):
    if i % 5 == 4:
        imp_knn_target_test.append(knn_target[i])
    else:
        imp_knn_target_train.append(knn_target[i])

# print(imp_knn_target_train)

# Lasso
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(rm_data_train, rm_target_train)
pred_train_lasso= model_lasso.predict(rm_data_train)
pred_test_lasso = model_lasso.predict(rm_data_test)
# print(mean_absolute_error(rm_target_test,pred_test_lasso))
# print(r2_score(rm_target_test, pred_test_lasso))

model_lasso.fit(imp_mean_train, imp_target_train)
pred_train_lasso= model_lasso.predict(imp_mean_train)
pred_test_lasso = model_lasso.predict(imp_mean_test)
# print(mean_absolute_error(imp_target_test,pred_test_lasso))
# print(r2_score(imp_target_test, pred_test_lasso))

model_lasso.fit(imp_knn_train, imp_knn_target_train)
pred_train_lasso= model_lasso.predict(imp_knn_train)
pred_test_lasso = model_lasso.predict(imp_knn_test)
# print(mean_absolute_error(imp_knn_target_test,pred_test_lasso))
# print(r2_score(imp_knn_target_test, pred_test_lasso))

model_lasso.fit(mice_train, mice_target_train)
pred_train_lasso= model_lasso.predict(mice_train)
pred_test_lasso = model_lasso.predict(mice_test)
# print(mean_absolute_error(mice_target_test,pred_test_lasso))
# print(r2_score(mice_target_test, pred_test_lasso))

# plt.scatter(mice_target_test, pred_test_lasso)
# plt.grid()
# plt.xlabel('Actual y')
# plt.ylabel('Predicted y')
# plt.title('Scatter plot between actual y and predicted y')
# plt.show()
