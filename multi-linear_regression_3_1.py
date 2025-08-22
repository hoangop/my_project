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

# implementing for
best = ['',0]
for p in allowed_factors:
    model  = smf.ols(formula='Density~'+p, data=train_fat).fit()
    #print(p, model.rsquared)
    if model.rsquared>best[1]:
        best = [p, model.rsquared]


# your code here
train_bmi1 = smf.ols(formula='Density~' + best[0], data=train_fat).fit()
print(train_bmi1.summary())

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
#print(train_bmi2.summary())
best_k2_predictors = [best_k1_predictor, best_k2_combo[0]]
remaining_factors_k3 = [p for p in allowed_factors if p not in best_k2_predictors]

best_k3_combo = ['', 0]
print("\nĐánh giá các mô hình với 3 biến dự đoán:")
for p3 in remaining_factors_k3:
    formula = f'Density ~ {" + ".join(best_k2_predictors)} + {p3}'
    model_k3 = smf.ols(formula=formula, data=train_fat).fit()
    print(f"Mô hình với {best_k2_predictors[0]}, {best_k2_predictors[1]} và {p3}: Adjusted R-squared = {model_k3.rsquared_adj:.4f}")
    if model_k3.rsquared_adj > best_k3_combo[1]:
        best_k3_combo = [p3, model_k3.rsquared_adj]

print(f"\nBiến thứ ba tốt nhất cho k=3 là: {best_k3_combo[0]}")
print(f"Tổ hợp tốt nhất cho k=3 là: {best_k2_predictors[0]}, {best_k2_predictors[1]} và {best_k3_combo[0]}")

# Lưu lại mô hình tốt nhất với 3 biến dự đoán
final_predictors_k3 = best_k2_predictors + [best_k3_combo[0]]
formula_k3 = f'Density ~ {" + ".join(final_predictors_k3)}'
train_bmi3 = smf.ols(formula=formula_k3, data=train_fat).fit()
# print(train_bmi3.summary())

# your code here
best_k3_predictors = final_predictors_k3
remaining_factors_k4 = [p for p in allowed_factors if p not in best_k3_predictors]

best_k4_combo = ['', 0]
print("\nĐánh giá các mô hình với 4 biến dự đoán:")
for p4 in remaining_factors_k4:
    formula = f'Density ~ {" + ".join(best_k3_predictors)} + {p4}'
    model_k4 = smf.ols(formula=formula, data=train_fat).fit()
    print(f"Mô hình với {best_k3_predictors[0]}, {best_k3_predictors[1]}, {best_k3_predictors[2]} và {p4}: Adjusted R-squared = {model_k4.rsquared_adj:.4f}")
    if model_k4.rsquared_adj > best_k4_combo[1]:
        best_k4_combo = [p4, model_k4.rsquared_adj]

print(f"\nBiến thứ tư tốt nhất cho k=4 là: {best_k4_combo[0]}")
print(f"Tổ hợp tốt nhất cho k=4 là: {best_k3_predictors[0]}, {best_k3_predictors[1]}, {best_k3_predictors[2]} và {best_k4_combo[0]}")

# Lưu lại mô hình tốt nhất với 4 biến dự đoán
final_predictors_k4 = best_k3_predictors + [best_k4_combo[0]]
formula_k4 = f'Density ~ {" + ".join(final_predictors_k4)}'
train_bmi4 = smf.ols(formula=formula_k4, data=train_fat).fit()
#print(train_bmi4.summary())

# your code here
best_k4_predictors = final_predictors_k4
remaining_factors_k5 = [p for p in allowed_factors if p not in best_k4_predictors]

best_k5_combo = ['', 0]
print("\nEvaluate the model with 5 predictors:")
for p5 in remaining_factors_k5:
    formula = f'Density ~ {" + ".join(best_k4_predictors)} + {p5}'
    model_k5 = smf.ols(formula=formula, data=train_fat).fit()
    print(f"Model with {best_k4_predictors[0]}, {best_k4_predictors[1]}, {best_k4_predictors[2]}, {best_k4_predictors[3]} and {p5}: Adjusted R-squared = {model_k5.rsquared_adj:.4f}")
    if model_k5.rsquared_adj > best_k5_combo[1]:
        best_k5_combo = [p5, model_k5.rsquared_adj]

print(f"\nThe best 5th predictor for k=5 is: {best_k5_combo[0]}")
print(f"The best combination for k=5 is: {best_k4_predictors[0]}, {best_k4_predictors[1]}, {best_k4_predictors[2]}, {best_k4_predictors[3]} and {best_k5_combo[0]}") 
print(f"with highest Adjusted R-squared is: {best_k5_combo[1]:.4f}")

# Lưu lại mô hình tốt nhất với 5 biến dự đoán
final_predictors_k5 = best_k4_predictors + [best_k5_combo[0]]
formula_k5 = f'Density ~ {" + ".join(final_predictors_k5)}'
train_bmi5 = smf.ols(formula=formula_k5, data=train_fat).fit()
print(train_bmi5.summary())


# plot resulting adjusted rsquared vs number of predictors (k=1,2,3,4,5)
# overlay the adjusted rsquared for the test data 
# your code here
# Extract Adjusted R-squared values from trained models
adjr2_train = [
    train_bmi1.rsquared_adj,
    train_bmi2.rsquared_adj,
    train_bmi3.rsquared_adj,
    train_bmi4.rsquared_adj,
    train_bmi5.rsquared_adj
]

# Compute R-squared for the test set
# Note: Adjusted R-squared is typically not used for test evaluation
# We compute R-squared instead for easier comparison
def calculate_r_squared(model, test_data):
    predictions = model.predict(test_data)
    # Compute R-squared by comparing residual sum of squares with total sum of squares
    ss_total = ((test_data['Density'] - test_data['Density'].mean()) ** 2).sum()
    ss_residual = ((test_data['Density'] - predictions) ** 2).sum()
    return 1 - (ss_residual / ss_total)

adjr2_test = [
    calculate_r_squared(train_bmi1, test_fat),
    calculate_r_squared(train_bmi2, test_fat),
    calculate_r_squared(train_bmi3, test_fat),
    calculate_r_squared(train_bmi4, test_fat),
    calculate_r_squared(train_bmi5, test_fat)
]

# print(f"\nadjr2_train: {adjr2_train}")
# print(f"adjr2_test: {adjr2_test}")

# Plot
k = [1, 2, 3, 4, 5]
plt.figure(figsize=(10, 6))
plt.plot(k, adjr2_train, marker='o', linestyle='-', color='blue', label='Train Adjusted R-squared')
plt.plot(k, adjr2_test, marker='o', linestyle='--', color='red', label='Test R-squared')
plt.title('Comparison of Adjusted R-squared on Training and Test Sets')
plt.xlabel('Number of predictors (k)')
plt.ylabel('R-squared / Adjusted R-squared')
plt.xticks(k)
plt.legend()
plt.grid(True)
plt.show()
