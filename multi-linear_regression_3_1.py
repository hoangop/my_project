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

#print(bmi.summary())

#3b
# Bước 5: Vẽ đồ thị tán xạ và đường hồi quy cho bài 3b
# Tạo một hình vẽ với kích thước phù hợp
# plt.figure(figsize=(10, 6))

# # Sử dụng seaborn để tạo biểu đồ tán xạ và đường hồi quy
# # Tán xạ biểu thị mối quan hệ thực tế giữa BMI và Density
# sns.scatterplot(x='BMI', y='Density', data=cfat, color='blue', label='Real Data')

# # Thêm đường hồi quy tuyến tính từ mô hình
# sns.regplot(x='BMI', y='Density', data=cfat, scatter=False, color='red', label='Regression line')

# # Gán nhãn cho các trục và tiêu đề biểu đồ
# plt.title('Relationship between BMI and Density with regression line')
# plt.xlabel('BMI Index (kg/m²)')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)

# # Hiển thị biểu đồ
# plt.show()

#3E
# your code here
allowed_factors = ['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm','Wrist']

from sklearn.model_selection import train_test_split
train_fat, test_fat = train_test_split(cfat, train_size=125, random_state=0)

best = ['',0]
for p in allowed_factors:
    model  = smf.ols(formula='Density~'+p, data=train_fat).fit()
    #print(p, model.rsquared)
    if model.rsquared>best[1]:
        best = [p, model.rsquared]

best_k1_predictor = best[0]
remaining_factors = [p for p in allowed_factors if p != best_k1_predictor]

best_k2_combo = ['', 0]
print("\nEvaluate the model with 2 predictors:")
for p2 in remaining_factors:
    formula = f'Density ~ {best_k1_predictor} + {p2}'
    model_k2 = smf.ols(formula=formula, data=train_fat).fit()
    print(f"Model with {best_k1_predictor} and {p2}: Adjusted R-squared = {model_k2.rsquared_adj:.4f}")
    if model_k2.rsquared_adj > best_k2_combo[1]:
        best_k2_combo = [p2, model_k2.rsquared_adj]
print(f"\nBest combination (The highest Adj. R-square) of predictors is {best_k1_predictor} and {best_k2_combo[0]} with Adjusted R-squared = {best_k2_combo[1]:.4f}")

train_bmi2 = smf.ols(formula=f'Density ~ {best_k1_predictor} + {best_k2_combo[0]}', data=train_fat).fit()
print(train_bmi2.summary())

