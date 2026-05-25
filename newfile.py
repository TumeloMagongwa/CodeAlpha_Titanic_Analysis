import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data - correct path
df = pd.read_csv('/storage/emulated/0/Download/archive/train_and_test2.csv')

# 2. TASK 2: EDA - Print structure and missing values
print("=== DATA STRUCTURE ===")
print(f"Shape: {df.shape}")
print("\nColumns and Data Types:")
print(df.dtypes)
print("\nMissing Values per Column:")
print(df.isnull().sum())

# 3. TASK 3: VISUAL 1 - Missing values heatmap
plt.figure(figsize=(10,6))
plt.imshow(df.isnull(), aspect='auto', cmap='viridis')
plt.title('Missing Values Heatmap')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.tight_layout()
plt.savefig('missing_heatmap.png', dpi=150, facecolor='white')
plt.close()

# 4. TASK 3: VISUAL 2 - Survival count bar chart
survived_col = df.columns[-1]  # gets the last column name
survived_counts = df[survived_col].value_counts().sort_index()

plt.figure(figsize=(6,4))
plt.bar(survived_counts.index.astype(str), survived_counts.values, color=['red','green'])
plt.title('Survival Count')
plt.xlabel('survived')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.savefig('survival_count.png', dpi=150, facecolor='white')
plt.close()

print("\nDone! Check Pydroid 3 > files folder for:")
print("1. missing_heatmap.png")
print("2. survival_count.png")