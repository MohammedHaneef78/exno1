# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
# Step 2: Read the Dataset
# Replace with your actual CSV file
df1 = pd.read_csv('Loan_data.csv')
df1.head()
# Step 3: Dataset Information
df1.info()
df1.describe()
# Step 4: Handling Missing Values
# Check Null Values
df1.isnull()
df1.isnull().sum()
# Fill Missing Values with 0
df1_fill_0 = df1.fillna(0)
df1_fill_0
# Forward Fill
df1_ffill = df1.ffill()
df1_ffill
# Backward Fill
df1_bfill = df1.bfill()
df1_bfill
# Fill with Mean (Numerical Column Example)
df1['CoapplicantIncome'] =
df1['CoapplicantIncome'].fillna(df1['CoapplicantIncome'].mean())
df1
# Drop Missing Values
df1_dropna = df1.dropna()
df1_dropna
#Step 5: Save Cleaned Data
df1_dropna.to_csv('clean_data.csv', index=False)
# OUTLIER DETECTION
# Step 6: IQR Method (Using Iris Dataset)
ir = pd.read_csv('iris.csv')
ir.head()
ir.info()
ir.describe()
#Boxplot for Outlier Detection
sns.boxplot(x=ir['sepal_width'])
plt.show()
# Calculate IQR
Q1 = ir['sepal_width'].quantile(0.25)
Q3 = ir['sepal_width'].quantile(0.75)
IQR = Q3 - Q1
print("IQR:", IQR)
# Detect Outliers
outliers_iqr = ir[
 (ir['sepal_width'] < (Q1 - 1.5 * IQR)) |
 (ir['sepal_width'] > (Q3 + 1.5 * IQR))
]
outliers_iqr
# Remove Outliers
ir_cleaned = ir[
 ~((ir['sepal_width'] < (Q1 - 1.5 * IQR)) |
 (ir['sepal_width'] > (Q3 + 1.5 * IQR)))
]
ir_cleaned
# Step 7: Z-Score Method
data = [1,12,15,18,21,24,27,30,33,36,39,42,45,48,51,
 54,57,60,63,66,69,72,75,78,81,84,87,90,93]
df_z = pd.DataFrame(data, columns=['values'])
df_z
# Calculate Z-Scores
z_scores = np.abs(stats.zscore(df_z))
z_scores
# Detect Outliers
threshold = 3
outliers_z = df_z[z_scores > threshold]
print("Outliers:")
outliers_z
# Remove Outliers
df_z_cleaned = df_z[z_scores <= threshold]
df_z_cleaned

<img width="666" height="547" alt="image" src="https://github.com/user-attachments/assets/5ca7f86e-6fee-4147-aae8-673cb11ba379" />


<img width="1032" height="239" alt="image" src="https://github.com/user-attachments/assets/f5f4a546-3e54-4988-a9ea-a89006d8811a" />


<img width="1088" height="877" alt="image" src="https://github.com/user-attachments/assets/d3e02500-7ad2-4185-8a4a-43bf976470f0" />


<img width="1074" height="315" alt="image" src="https://github.com/user-attachments/assets/d5a9e316-73ac-4ca6-a581-9eb860051f64" />

<img width="1083" height="903" alt="image" src="https://github.com/user-attachments/assets/60d950e3-4482-4d05-91e3-6cb68ee04735" />

<img width="1030" height="394" alt="image" src="https://github.com/user-attachments/assets/dc502047-4dbc-4508-a2ce-4930febd82c5" />

<img width="1086" height="915" alt="image" src="https://github.com/user-attachments/assets/4e2a8b3d-a05d-487f-918c-a3b90cabb028" />

<img width="1092" height="315" alt="image" src="https://github.com/user-attachments/assets/58027114-6dfa-4ac1-90a2-a5d818eb26cb" />

   
# Result
Hence the data was cleaned.
         
