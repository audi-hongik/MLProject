
import numpy as np
import os
import pandas as pd  # data processing, CSV file I/O

data = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
# preview the first 5 lines of the loaded data
print(data)

# NAN값 있는 행 제거
# data_re = data.dropna(axis=0)
# print(data_re)

# Name, NA_Sales, EU_Sales, JP_Sales, Other_Sales 제외
# Global_Sales 는 output

# anxis =1 은 열을 삭제하겠다는 것을 뜻한다
data.drop(['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1, inplace=True)
print(data)