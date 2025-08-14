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

# #2c
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
print(model_multi.summary())