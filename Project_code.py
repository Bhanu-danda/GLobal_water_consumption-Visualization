import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


#loading the dataset
df=pd.read_csv(r"C:\Users\BHANU PRASAD\Desktop\SEM 4\Python Int 375\Global water consumption python - Ca-2\global_water_consumption.csv")

# Basic Info
print("Basic Info:")
print(df.info())
print("\n")

# First few rows
print("Head of the DataFrame:")
print(df.head())

# Summary statistics
print("\nDescriptive Statistics:")
print(df.describe())

# check for missing values  in the coloumns and the total dataset
print(df.isnull().sum()) 
total_missing = df.isnull().sum().sum()
print(f"Total missing values: {total_missing}")


# Replaceing missing values with column mean
df.dropna(inplace=True)
df = df.fillna(df.mean(numeric_only=True))   #numeric  
df.fillna("Unknown", inplace=True)         #non-numeric


##----------------------------------------------------------------- OBJECTIVES -------------------------------------------###

# Objective 1: Compare Total Water Consumption by Country
df = df.drop_duplicates(subset='Country')  
df = df.dropna(subset=['Total Water Consumption (Billion Cubic Meters)'])  
top_n = 10
top_countries = df.nlargest(top_n, 'Total Water Consumption (Billion Cubic Meters)')
plt.figure(figsize=(12, 6))
sns.barplot(data=top_countries, x='Country', y='Total Water Consumption (Billion Cubic Meters)', hue='Country', palette='Blues_r', legend=False)
plt.title('Total Water Consumption by Country', fontsize=14, color='navy')
plt.xlabel('Country', color='darkblue')
plt.ylabel('Total Water Consumption (Billion m³)', color='darkblue')
plt.xticks(rotation=45, ha='right', color='black')  
plt.yticks(color='black')
plt.tight_layout()
plt.show()

# Objective 2: Proportion of Water Use by Sector per Country
water_use = df[['Country', 'Agricultural Water Use (%)', 'Industrial Water Use (%)', 'Household Water Use (%)']]
water_use.set_index('Country', inplace=True)
colors = ['#FF6347', 'lightblue', 'blue']  
water_use.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6))
plt.title('Sector-wise Water Use by Country', fontsize=14, color='navy')
plt.xlabel('Country', color='darkblue')
plt.ylabel('Water Use (%)', color='darkblue')
plt.xticks(color='black')
plt.yticks(color='black')
plt.legend(title='Sector')
plt.tight_layout()
plt.show()

# Objective 3: Water Scarcity Level 
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='Water Scarcity Level', hue='Water Scarcity Level', palette='Blues', legend=False)
plt.title('Water Scarcity Level Distribution', fontsize=14, color='navy')
plt.xlabel('Water Scarcity Level', color='darkblue')
plt.ylabel('Number of Countries', color='darkblue')
plt.xticks(color='black')
plt.yticks(color='black')
plt.tight_layout()
plt.show()


#  Objective 4: Correlation Heatmap for Numerical Features
plt.figure(figsize=(10,10))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Between Water Indicators', fontsize=14, color='navy')
plt.xticks(color='black')
plt.yticks(color='black')
plt.tight_layout()
plt.show()

#Objective 5: Groundwater Depletion Rate by Country
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Country', y='Groundwater Depletion Rate (%)', hue='Country', palette='Blues_d', legend=False)
plt.title('Groundwater Depletion Rate by Country', fontsize=14, color='navy')
plt.xlabel('Country', color='darkblue')
plt.ylabel('Groundwater Depletion Rate (%)', color='darkblue')
plt.xticks(rotation=45, color='black')
plt.yticks(color='black')
plt.tight_layout()
plt.show()


#Objective 6: Box Plot of Water Use by Sector (Agricultural, Industrial, Household)
plt.figure(figsize=(7, 7))
sns.boxplot(data=df[['Agricultural Water Use (%)', 'Industrial Water Use (%)', 'Household Water Use (%)']], palette='Blues')
plt.title('Distribution of Water Use by Sector Across Countries', fontsize=14, color='navy')
plt.xlabel('Water Use Sector', color='darkblue')
plt.ylabel('Water Use (%)', color='darkblue')
plt.xticks(color='black')
plt.yticks(color='black')
plt.tight_layout()
plt.show()

# Objective 7: Water Use Proportions for Each Country (Pie Chart)
country_data = df[df['Country'] == 'India'][['Agricultural Water Use (%)', 'Industrial Water Use (%)', 'Household Water Use (%)']]
plt.figure(figsize=(8, 8))
plt.pie(country_data.values[0], labels=country_data.columns, autopct='%1.1f%%', colors=['#FF6347', 'lightblue', 'blue'], startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('Water Use Distribution by Sector in India', fontsize=14, color='navy')
plt.tight_layout()
plt.show()


#-------------------------------------------------------------- T-Test on dataset ----------------------------------------------------#
# Group based on actual values
scarce = df[df['Water Scarcity Level'] == 'High']
non_scarce = df[df['Water Scarcity Level'] == 'Low']

# Print group sizes
print("\n📊 Group Sizes:")
print(f"Water-Scarce Countries (High): {scarce.shape[0]}")
print(f"Non-Water-Scarce Countries (Low): {non_scarce.shape[0]}")

# Check if both groups are valid
if scarce.shape[0] >= 2 and non_scarce.shape[0] >= 2:
    t_stat, p_value = ttest_ind(
        scarce['Per Capita Water Use (Liters per Day)'].dropna(),
        non_scarce['Per Capita Water Use (Liters per Day)'].dropna(),
        equal_var=False  # Welch's t-test
    )

    print("\n--- T-Test: Per Capita Water Use (High vs Low Scarcity) ---")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Significant difference in per capita water use.")
    else:
        print("Result: No significant difference found.")
else:
    print("\nNot enough data in one or both groups to perform a valid T-test.")





