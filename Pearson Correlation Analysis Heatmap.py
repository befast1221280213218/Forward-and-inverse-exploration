import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read file
file_path = r'******'
df = pd.read_excel(file_path)

# Normalize the data
A = ** 
df['Elastic modulus'] = df['Elastic modulus'] / A
df['yield strength'] = df['yield strength'] / A
df['Tensile strength'] = df['Tensile strength'] / A 
df['Toughness'] = df['Toughness'] / A
df['Structure'] = df['Structure'] / A
df['Dilameter'] = df['Dilameter'] / A
df['Volume'] = df['Volume'] / A
df['Variance'] = df['Variance'] / A

# Select necessary columns
selected_columns = ['Elastic modulus', 'yield strength', 'Tensile strength', 'Toughness', 'Structure', 'Dilameter', 'Volume', 'Variance']

# Calculate Pearson correlation matrix
correlation_matrix = df[selected_columns].corr()

# Plot heatmap
plt.figure(figsize=(30, 20))
sns.set(font_scale=3)  # Set font size
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size":35})
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=35)  # Set colorbar font size
plt.title('Pearson Correlation Heatmap', fontsize=40)  # Adjusted title font size
plt.show()
