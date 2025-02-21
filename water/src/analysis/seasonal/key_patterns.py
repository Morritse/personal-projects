import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(20, 10))

# Read data
df = pd.read_csv('water_years_base.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
spills = df[df['Reservoir_Level'] > 522500].copy()

# Create subplots
plt.subplot(1, 2, 1)
# Monthly spill distribution
monthly_spills = spills.groupby('Month').size()
sns.barplot(x=monthly_spills.index, y=monthly_spills.values)
plt.title('Number of Spill Days by Month')
plt.xlabel('Month')
plt.ylabel('Number of Spill Days')
for i, v in enumerate(monthly_spills.values):
    if v > 0:
        plt.text(i, v, str(int(v)), ha='center', va='bottom')

plt.subplot(1, 2, 2)
# Inflow distribution during spills vs non-spills
sns.boxplot(data=df, x='Water_Year_Type', y='Inflow', 
            hue=df['Reservoir_Level'] > 520000)
plt.title('Inflow Distribution by Year Type\n(Spill vs Non-Spill Days)')
plt.xlabel('Water Year Type')
plt.ylabel('Inflow')
plt.legend(title='Spill Day', labels=['No', 'Yes'])

plt.tight_layout()
plt.savefig('key_patterns.png')

# Print detailed statistics
print("\nDetailed Statistics:")
print("\n1. Spill Days by Water Year Type and Month:")
pivot = pd.pivot_table(spills, index='Water_Year_Type', columns='Month', 
                      values='Reservoir_Level', aggfunc='count', fill_value=0)
print(pivot)

print("\n2. Average Inflow During Spills by Year Type:")
spill_inflow = spills.groupby('Water_Year_Type')['Inflow'].agg(['mean', 'count'])
print(spill_inflow.sort_values('count', ascending=False))
