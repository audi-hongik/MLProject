from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer  # imputation library
from impyute.imputation.cs import mice, fast_knn  # multiple imputation librar
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score


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
data_remove = original_data.dropna(axis=0)
original_data = original_data.replace('tbd', np.NaN)


# 1.5 String인 컬럼 변환
original_data = original_data.astype({'User_Score': 'float64'})

# 1.6 원핫 인코딩으로 라벨링
categorical_subset = original_data[["Platform", "Genre", "Publisher", "Rating"]]
encoder = ce.one_hot.OneHotEncoder()
categorical_subset = encoder.fit_transform(categorical_subset)
categorical_subset = categorical_subset.reset_index()


datasets = {}

# 방법1> KNN imputer
imp_knn = KNNImputer(n_neighbors=7)
numeric_subset = original_data[["Year_of_Release", "Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]]
imp_knn = imp_knn.fit_transform(numeric_subset)
imp_knn = pd.DataFrame(imp_knn)
data_knn = pd.concat([imp_knn, categorical_subset], axis = 1)
datasets["data_knn"] = data_knn

# 방법2> Multiple imputer
data_subset = original_data[["Year_of_Release", "Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]]
imp_mice = mice(data_subset.to_numpy())
imp_mice = pd.DataFrame(imp_mice)
data_mice = pd.concat([imp_mice, categorical_subset], axis = 1)
datasets["data_mice"] = data_mice

# 2. 데이터 분리
for key, imputed_data in datasets.items():
    print("****"+key+"****")

    # 2.1 data와 target 분리
    if key == "data_null":
        data = imputed_data.drop("Global_sales", axis=1).values
        target = imputed_data["Global_sales"].values
    else:
        data = imputed_data.drop(1, axis=1).values
        target = imputed_data[1].values

    # 2.2 test data와 training data 분리
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)

    # 3. knn regression
    print("KNN regression")
    regressor = KNeighborsRegressor(7, "distance")
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    mae_knn = mae(y_test, y_pred)
    print("MAE: ", mae_knn)
    print("R^2: ", r2_score(y_test, y_pred))

    # 4. Linear Regression
    print("Linear Regression")
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)

    y_predict = mlr.predict(x_test)
    mae_mlr = mae(y_test, y_predict)
    print("MAE: ", mae_mlr)
    print("R^2: ", r2_score(y_test, y_predict))

    # 5. Lasso Regression
    print("Lasso Regression")
    model_lasso = Lasso(alpha=0.01)
    model_lasso.fit(x_train, y_train)

    pred_test_lasso = model_lasso.predict(x_test)
    print("MAE: ", mean_absolute_error(y_test, pred_test_lasso))
    print("R^2: ", r2_score(y_test, pred_test_lasso))