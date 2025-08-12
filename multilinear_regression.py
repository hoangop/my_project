
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

columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
df = pd.read_csv("data/auto-mpg/auto-mpg.data", header=None, delimiter=r"\s+", names=columns) 
#print(df.info())
#print(df.describe())

# Data Cleaning Steps
# fix data types for 'horsepower'
# 1. Convert 'horsepower' to numeric, forcing errors to NaN
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# 2. Remove rows with NaN values
# Now that non-numeric values are converted to NaN, we can drop them.
# The `dropna()` function will remove any rows with missing values.
df = df.dropna()

# 3. Drop the 'car_name' column

df = df.drop('car_name', axis=1)

# corr_matrix = df.corr()

# plt.figure(figsize=(18, 16))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": .8})
# plt.title('Correlation Heatmap', fontsize=20)
# plt.show()

# model = smf.ols(formula='mpg ~ acceleration', data=df).fit()

# # # In ra bản tóm tắt kết quả của mô hình
# print("\nThe result of model sumary:")
# print(model.summary())

# --- Code for part 1d ---
# Normalize the 'weight' feature
df['weight_norm'] = df['weight'] / df['weight'].mean()

# Initialize variables to track the best result
best_degree = 1
best_r_squared = 0.0

# Initialize variable for the "highest order that makes sense"
# This value needs to be determined by inspecting the model summary for each degree.
# A higher R-squared doesn't always mean a better model if it's overfitting.
# We will leave this for the user to determine after running the code.
sound_degree = 1

# Print header for output
print("--- Calculating R-squared for polynomial degrees 1 to 20 on 'weight_norm' ---")
print("----------------------------------------------------------------------------")

# Loop through polynomial degrees from 1 to 20
for n in range(1, 21):
    # Dynamically build the formula string for statsmodels
    if n == 1:
        # Base case for a linear model
        formula = "mpg ~ weight_norm"
    else:
        # For n > 1, build the formula with np.power() for each term
        formula_parts = [f'np.power(weight_norm, {i})' for i in range(2, n + 1)]
        formula = f"mpg ~ weight_norm + {' + '.join(formula_parts)}"
    
    try:
        # Fit the Ordinary Least Squares (OLS) model
        model = smf.ols(formula, data=df).fit()
        
        # Get the R-squared value from the model summary
        r_squared = model.rsquared
        p_value = model.pvalues['weight_norm']
        
        # Print the R-squared for the current degree
        print(f"Polynomial Degree {n:2}: R-squared = {r_squared:.4f} P_value = {p_value: 4f} " )
        
        # Optionally, you can uncomment this line to inspect the full summary
        # print(model.summary())
        
        # Check if this degree gives a better R-squared value
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_degree = n
            
    except Exception as e:
        print(f"Error fitting model for degree {n}: {e}")

# Print the final result after the loop
print("\n------------------------------------------------------------")
print("--- Best Result Summary based on R-squared ---")
print(f"The best polynomial degree is {best_degree} with an R-squared of {best_r_squared:.4f}")
print("\n--- Additional Analysis ---")
print("What do you observe from the results? How does normalizing the feature affect the R-squared values?")
print("Inspect the model summaries to determine the 'sound_degree' (the highest order model that makes sense).")
print("The current value for 'sound_degree' is: ", sound_degree)
print("------------------------------------------------------------")
