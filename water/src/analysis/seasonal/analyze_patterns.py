import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(20, 15))

# Read data
df = pd.read_csv('water_years_base.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Create subplots
plt.subplot(2, 2, 1)
# Average reservoir level by month for each water year type
monthly_avg = df.groupby(['Water_Year_Type', 'Month'])['Reservoir_Level'].mean().unstack()
sns.heatmap(monthly_avg, cmap='YlOrRd', annot=True, fmt='.0f')
plt.title('Average Reservoir Level by Month and Year Type')
plt.xlabel('Month')
plt.ylabel('Water Year Type')

plt.subplot(2, 2, 2)
# Relationship between inflow and reservoir level
sns.scatterplot(data=df, x='Inflow', y='Reservoir_Level', hue='Water_Year_Type', alpha=0.5)
plt.axhline(y=522500, color='red', linestyle='--', label='Spill Level')
plt.title('Inflow vs Reservoir Level by Year Type')
plt.legend(title='Water Year Type')

plt.subplot(2, 2, 3)
# Distribution of reservoir levels by water year type
sns.boxplot(data=df, x='Water_Year_Type', y='Reservoir_Level')
plt.axhline(y=522500, color='red', linestyle='--', label='Spill Level')
plt.title('Reservoir Level Distribution by Year Type')
plt.legend()

plt.subplot(2, 2, 4)
# Monthly spill occurrence
spills = df[df['Reservoir_Level'] > 522500].copy()
spill_counts = spills.groupby(['Water_Year_Type', 'Month']).size().unstack(fill_value=0)
sns.heatmap(spill_counts, cmap='YlOrRd', annot=True, fmt='d')
plt.title('Number of Spill Days by Month and Year Type')
plt.xlabel('Month')
plt.ylabel('Water Year Type')

plt.tight_layout()
plt.savefig('reservoir_patterns.png')

# Print some interesting statistics
print("\nInteresting Patterns:")
print("\n1. Months with most spills:")
monthly_spills = spills.groupby('Month').size().sort_values(ascending=False)
print(monthly_spills.head())

print("\n2. Average reservoir level by year type:")
print(df.groupby('Water_Year_Type')['Reservoir_Level'].mean().sort_values(ascending=False))

print("\n3. Year types with most spills:")
print(spills.groupby('Water_Year_Type').size().sort_values(ascending=False))

print("\n4. Average inflow during spills:")
print(f"During spills: {spills['Inflow'].mean():.2f}")
print(f"Overall: {df['Inflow'].mean():.2f}")
