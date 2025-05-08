import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('heart.csv')

print("Dataset shape:", df.shape)
print("\nDataset columns:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nNumerical columns ({len(numerical_cols)}):", numerical_cols)

min_max_values = pd.DataFrame({
    'Min': df[numerical_cols].min(),
    'Max': df[numerical_cols].max()
})
print("\nMin and Max values for each numerical column:")
print(min_max_values)

plt.figure(figsize=(14, 8))

x = np.arange(len(numerical_cols))
width = 0.35

plt.bar(x - width/2, min_max_values['Min'], width, label='Min', color='skyblue', edgecolor='darkblue', linewidth=1)
plt.bar(x + width/2, min_max_values['Max'], width, label='Max', color='orange')

plt.xlabel('Numerical Columns')
plt.ylabel('Values')
plt.title('Minimum and Maximum Values for Numerical Columns in Heart Dataset')
plt.xticks(x, numerical_cols, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('heart_data_min_max.png')

plt.show()

