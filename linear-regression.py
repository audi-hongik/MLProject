from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer  # imputation library
from impyute.imputation.cs import mice, fast_knn  # multiple imputation librar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDRegressor


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

# 1.7 String인 컬럼 변환
original_data = original_data.astype({'User_Score': 'float64'})

# 1.5 Null값의 보정
imputer_knn = KNNImputer(n_neighbors=7)
numeric_subset = original_data[
    ["Year_of_Release", "Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]]
data = numeric_subset.to_numpy()
imputed_knn = imputer_knn.fit_transform(numeric_subset)
imputed_knn = pd.DataFrame(imputed_knn)

# 1.6 원핫 인코딩으로 라벨링
categorical_subset = original_data[["Platform", "Genre", "Publisher", "Rating"]]
encoder = ce.one_hot.OneHotEncoder()
categorical_subset = encoder.fit_transform(categorical_subset)
categorical_subset = categorical_subset.reset_index()
features = pd.concat([imputed_knn, categorical_subset], axis=1)

# 2. 데이터 분리
# 2.1 data와 target 분리
data = features.drop(1, axis=1).values
target = features[1].values

# 2.2 test data와 training data 분리
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)


class GradientDescent():
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=1000):
        self._learning_rate = learning_rate
        self._threshold = threshold
        self._max_iterations = max_iterations
        self._W = None

    def fit(self, x_data, y_data):
        num_examples, num_features = np.shape(x_data)
        self._W = np.ones(num_features)
        x_data_transposed = x_data.transpose()

        for i in range(self._max_iterations):
            # 실제값과 예측값의 차이
            diff = np.dot(x_data, self._W) - y_data

            # diff를 이용하여 cost 생성 : 오차의 제곱합 / 2 * 데이터 개수
            cost = np.sum(diff ** 2) / (2 * num_examples)

            # transposed X * cost / n
            gradient = np.dot(x_data_transposed, diff) / num_examples

            # W벡터 업데이트
            self._W = self._W - self._learning_rate * gradient

            # 판정 임계값에 다다르면 학습 중단
            if cost < self._threshold:
                return self._W

        return self._W

gd = GradientDescent()
print(gd.fit(x_train, y_train))

####################
# 3. linear regression
# W = np.random.rand(375, 1)
# b = np.random.rand(1)
#
# def loss_func(x, t):
#     # Y = X * W + b
#     y = np.dot(x, W) + b
#
#     # 각 오차들의 제곱의 합 평균
#     return np.sum((t - y) ** 2) / len(x)
#
# def numerical_derivative(fx, input_list):
#     delta_x = 1e-4
#
#     ret = np.zeros_like(input_list)
#     it = np.nditer(input_list, flags=['multi_index'], op_flags=['readwrite'])
#     while not it.finished:
#         i = it.multi_index
#
#         tmp = input_list[i]
#         input_list[i] = float(tmp) - delta_x
#         f1 = fx(input_list)
#
#         input_list[i] = float(tmp) + delta_x
#         f2 = fx(input_list)
#         ret[i] = (f2 - f1) / (delta_x * 2)
#
#         input_list[i] = tmp
#         it.iternext()
#
#     return ret
#
#
# # 오차를 알려주는 함수
# def error_val(x, t):
#     y = np.dot(x, W) + b
#     return np.sum((t - y) ** 2) / len(x)
#
#
# # 학습 이후에 예측해주는 함수
# def predict(x):
#     y = np.dot(x, W) + b
#     return y
#
# 학습 가중치
# learning_rate = 1e-5
#
# f = lambda x: loss_func(x_train, y_train)
# print("Initial error value = ", error_val(x_train, y_train))
#
# for step in range(2001):
#     # 학습
#     W -= learning_rate * numerical_derivative(f, W)
#     b -= learning_rate * numerical_derivative(f, b)
#
#     if step % 10 == 0:
#         print("step = ", step, "error value = ", error_val(x_train, y_train))
##########################
# W = np.random.rand(375, 1)
# b = 0.0
#
# n_data = len(x_train)
#
# epochs = 5000
# learning_rate = 0.01
#
# for i in range(epochs):
#     hypothesis = np.dot(x_train, W) + b
#     cost = np.sum((hypothesis - y_train) ** 2) / n_data
#     gradient_w = np.sum(np.dot((np.dot(x_train, W) - y_train + b) * 2 , x_train)) / n_data
#     gradient_b = np.sum((np.dot(x_train, W) - y_train + b) * 2) / n_data
#
#     W -= learning_rate * gradient_w
#     b -= learning_rate * gradient_b
#
#     if i % 100 == 0:
#         print('Epoch ({:10d}/{:10d}) cost: {:10f}, b:{:10f}'.format(i, epochs, cost, b))
