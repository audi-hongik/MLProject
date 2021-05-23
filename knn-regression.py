from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer  # imputation library
from impyute.imputation.cs import mice, fast_knn  # multiple imputation library

def mae(real_target, pred_target):
    return np.average(abs(real_target - pred_target))

# 1. 데이터 보정
# 1.1 데이터 불러오기
original_data = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")

# 1.2 중요한 column에 대해서 null값 행 아예 제거
original_data = original_data[original_data["Year_of_Release"].notnull()]
original_data = original_data[original_data["Genre"].notnull()]
original_data = original_data[original_data["Publisher"].notnull()]
original_data = original_data[original_data["Rating"].notnull()]

# 1.3 필요없는 컬럼 제거
original_data.drop(['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Developer'], axis=1, inplace=True)

# 1.4 tbd값 있는 행 제거
# data_remove = original_data.dropna(axis=0)
# data_remove = original_data.astype({'User_Score': 'float64'})
original_data = original_data.replace('tbd', np.NaN)

# 1.5 Null값의 보정
imputer_knn = KNNImputer(n_neighbors=5)
numeric_subset = original_data[["Year_of_Release", "Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]]
imputed_knn = imputer_knn.fit_transform(numeric_subset)
imputed_knn = pd.DataFrame(imputed_knn)

# 1.6 원핫 인코딩으로 라벨링
categorical_subset = original_data[["Platform", "Genre", "Publisher", "Rating"]]
encoder = ce.one_hot.OneHotEncoder()
categorical_subset = encoder.fit_transform(categorical_subset)
categorical_subset = categorical_subset.reset_index()
features = pd.concat([imputed_knn, categorical_subset], axis = 1)

# 1.7 String인 컬럼 변환
data_remove = original_data.astype({'User_Score': 'float64'})
#
# # 1.4 Platform, Genre, Rating에 대해서 one-hot encoding으로 변환
# #print(data_remove["Platform"].unique(), data_remove["Genre"].unique(), data_remove["Rating"].unique())
# numeric_subset = data_remove[[
#     "1", "5", "6", "7", "8", "4"]]
# categorical_subset = data_remove[["0", "2", "3", "9"]]
#
# encoder = ce.one_hot.OneHotEncoder()
# categorical_subset = encoder.fit_transform(categorical_subset)
#
# features = pd.concat([numeric_subset, categorical_subset], axis = 1)


# 2. 데이터 분리
# 2.1 data와 target 분리
data = features.drop(1, axis=1).values
target = features[1].values

# 2.2 test data와 training data 분리
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)

# 3. knn regression
regressor = KNeighborsRegressor(7, "distance")
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
model_mae = mae(y_test, y_pred)
print(y_test)
print(y_pred)

print(model_mae)