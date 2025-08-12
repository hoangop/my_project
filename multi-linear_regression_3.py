
import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set color map to have light blue background
sns.set()
import statsmodels.formula.api as smf
import statsmodels.api as sm

fat = pd.read_csv('data/bodyfat.csv')
fat = fat.drop('Unnamed: 0', axis=1)
fat.Weight = fat.Weight * 0.453592 # Convert to Kg
fat.Height = fat.Height * 0.0254 # convert inches to m
fat['BMI'] = fat.Weight / (fat.Height**2)

#fat.BMI.plot.kde();
#3a
cfat = fat[fat['BMI'] <= 40].copy()

bmi = smf.ols('Density ~ BMI', data=cfat).fit()

print(bmi.summary())

#3b
# Bước 5: Vẽ đồ thị tán xạ và đường hồi quy cho bài 3b
# Tạo một hình vẽ với kích thước phù hợp
plt.figure(figsize=(10, 6))

# Sử dụng seaborn để tạo biểu đồ tán xạ và đường hồi quy
# Tán xạ biểu thị mối quan hệ thực tế giữa BMI và Density
sns.scatterplot(x='BMI', y='Density', data=cfat, color='blue', label='Real Data')

# Thêm đường hồi quy tuyến tính từ mô hình
sns.regplot(x='BMI', y='Density', data=cfat, scatter=False, color='red', label='Regression line')

# Gán nhãn cho các trục và tiêu đề biểu đồ
plt.title('Mối quan hệ giữa BMI và Density với đường hồi quy')
plt.xlabel('Chỉ số BMI (kg/m²)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

# Hiển thị biểu đồ
plt.show()

