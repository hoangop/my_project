import scipy as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
# Set color map to have light blue background
sns.set()
import statsmodels.formula.api as smf
import statsmodels.api as sm
# %matplotlib inline  # Uncomment this line only when running in a Jupyter notebook


df = pd.read_csv('kc_house_data.csv')
#df.head()
#print(df.date)
#print(type(df.date.iloc[0]))

# extract year and month info from the string
# create new features 'sales_year' and 'sales_month' in df

#df['sales_year'] = df.date.apply(lambda x: int(x[:4]))
#df['sales_month'] = df.date.apply(lambda x: int(x[4:6]))
#print(df.groupby('sales_month')['id'].count())
#print(df.groupby('sales_year')['id'].count())


# your code here
# uncomment below and update the value as an integer
# sort_most_sales_month = df.groupby('sales_month')['id'].count().sort_values(ascending=False)
# sort_most_sales_year = df.groupby('sales_year')['id'].count().sort_values(ascending=False)
# most_sales_month = sort_most_sales_month.index[0]  # Get the month with the most sales
# most_sales_year = sort_most_sales_year.index[0]  # Get the year with the most sales
# your code here
# uncomment below and update the value as an integer
#least_sales_month = df.groupby('sales_month')['id'].count()

#print(f"Most sales month: {most_sales_month}")
#print(f"Most sales year: {most_sales_year}")
columns_to_drop = ['id', 'date', 'zipcode']

# Sử dụng .drop() để bỏ các cột.
# 'axis=1' chỉ định rằng chúng ta muốn bỏ các cột (thay vì các hàng).
# 'inplace=True' sẽ thay thế DataFrame ban đầu.
# Hoặc, một cách tốt hơn là gán kết quả trả về cho df để tránh lỗi và dễ đọc hơn.
# df.drop(columns_to_drop, axis=1, inplace=True)
df = df.drop(columns=columns_to_drop)

# In ra thông tin của DataFrame mới để xác nhận
#print(df.info())

# corr_matrix = df.corr()

# plt.figure(figsize=(18, 16))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": .8})
# plt.title('Correlation Heatmap', fontsize=20)
# plt.show()

# Chúng ta sẽ chia toàn bộ DataFrame df thành X_train và X_test
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# # In ra số lượng hàng của mỗi tập để kiểm tra
# print("\nSố lượng quan sát trong tập X_train:", len(X_train))
# print("Số lượng quan sát trong tập X_test:", len(X_test))

# # Kiểm tra xem 'price' có còn trong cả hai tập dữ liệu hay không
# print("\nCác cột trong X_train:", X_train.columns)
# print("Các cột trong X_test:", X_test.columns)
# # Testing cell for self-check
# assert(len(X_train) == 17290), "Check 3a, did you split properly so X_Train is 80% of the observations?"
# assert(type(X_train)==type(pd.DataFrame())), "Check 3a, what type of object should X_train be?"
# 3b) Train a simple linear regression model [5 pts]
# use best_guess_predictor as a single predictor
# build a simple linear regression model, train on the X_train portion

# Giả định 'best_guess_predictor' là 'sqft_living' vì đây là một trong những biến có tương quan cao nhất với 'price'.
model = smf.ols(formula='price ~ sqft_living', data=X_train).fit()

# # In ra bản tóm tắt kết quả của mô hình
print("\nThe result of model sumary:")
print(model.summary())


# # self test
# assert len(model.params.index) == 2, 'Check 3b, Number of model parameters (including intercept) does not match. Did you make a univariate model?'

# your code here

# uncomment and update top_three
#top_three = ['sqft_living','sqft_living','sqft_above']
# self test cell
# assert(type(top_three) == list), "Check 3c, the top_three needs to be a list."
# assert(len(top_three) == 3), "Check 3c, the top_three list needs to have three element."