
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

votes = pd.read_csv('data/fl2000.txt', delim_whitespace=True, comment='#')
votes = votes[['county', 'Bush', 'Gore', 'Nader', 'Buchanan']]
#votes.describe(include='all')

#2a
# numeric_votes = votes._get_numeric_data()
# corr_matrix = numeric_votes.corr()

# plt.figure(figsize=(18, 16))
# sns.pairplot(votes.iloc[:, :10], diag_kind='kde')
# plt.savefig('pair_plot.png', dpi = 300, bbox_inches = 'tight')

#2c
# y = votes['Bush']
# X = votes.drop(columns=['county', 'Bush'])

# # Add a constant term for the intercept
# X = sm.add_constant(X)
# model = sm.OLS(y, X).fit()
# print(model.summary())

#2d
# Create a formula for the multiple linear regression model
independent_vars = votes.drop(columns=['county', 'Bush']).columns
# Create formular with full interactions
#formula_str = 'Bush ~ ' + ' * '.join(independent_vars)
# Fit the model using statsmodels
#model_multi = smf.ols(formula=formula_str, data=votes).fit()
#Print and observe the p-value for each interaction
#print(model_multi.summary())

# Create formular eliminate interactions that do not meet the p value <= 0.05
formula_str = 'Bush ~ Gore +  Nader + Buchanan + Nader:Buchanan'
model_multi = smf.ols(formula=formula_str, data=votes).fit()
#print(model_multi.summary())

#2e
# leverage & square of residual calculation
leverage = model_multi.get_influence().hat_matrix_diag
residual_sq = model_multi.get_influence().resid_studentized_internal**2

# # Visualize leverage vs. square of residual
# plt.figure(figsize=(10, 6))
# plt.scatter(leverage, residual_sq, alpha=0.7)
# plt.axhline(y=3, color='r', linestyle='--', label='Cook\'s Distance')
# plt.title('Leverage vs. Residual Squared')
# plt.xlabel('Leverage')
# plt.ylabel('Square of the residual')
# plt.legend()
# plt.grid(True)
# plt.savefig('plot_leverage_residual.png', dpi = 300, bbox_inches = 'tight')

#2f
# Xác định các điểm bất thường (unusual)
# Tìm các điểm có leverage cao hoặc residual squared cao
# Leverage cao: Lớn hơn 2 * (p+1) / n, với p=13 (số biến) và n=67 (số quan sát)
# (Đây là một heuristic phổ biến. p ở đây là số biến độc lập trong công thức)
# residual_sq cao: Lớn hơn 3 (đường Cook's distance)
num_predictors = len(model_multi.params) - 1 # Bỏ đi intercept
leverage_threshold = 2 * num_predictors / len(votes)
residual_sq_threshold = 3

high_leverage_indices = np.where(leverage > leverage_threshold)[0]
high_residual_indices = np.where(residual_sq > residual_sq_threshold)[0]

# Gộp các chỉ mục (indices) lại và loại bỏ các giá trị trùng lặp
unusual = np.unique(np.concatenate([high_leverage_indices, high_residual_indices])).tolist()

# print("\n--------------------------------------------------------------------")
# print("Danh sách các chỉ mục của các điểm bất thường (unusual):", unusual)
# print("--------------------------------------------------------------------")

# # Hiển thị các dòng dữ liệu tương ứng
# print("\nThông tin chi tiết về các điểm bất thường:")
print(votes.loc[unusual])
# print(high_leverage_indices)
# print(high_residual_indices)

#2g
# Loại bỏ các điểm bất thường (unusual) khỏi DataFrame
votes_final = votes.drop(unusual, axis=0)

# Xây dựng mô hình hồi quy tuyến tính bội cuối cùng với dữ liệu đã làm sạch
# Tên biến của mô hình là 'model_final' theo yêu cầu
model_final = smf.ols(formula=formula_str, data=votes_final).fit()

# In ra bảng tóm tắt kết quả của mô hình cuối cùng
print(model_final.summary())
