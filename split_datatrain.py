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

corr_matrix = df.corr()

plt.figure(figsize=(18, 16))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": .8})
# plt.title('Correlation Heatmap', fontsize=18)
# plt.show()  

sns.pairplot(df.iloc[:, :10], diag_kind='kde')
plt.show()