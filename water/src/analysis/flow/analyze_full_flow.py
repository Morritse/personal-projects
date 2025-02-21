import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")

# Read data
df = pd.read_csv('water_years_base.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Create the plot
plt.figure(figsize=(15, 12))

# Scatter plot of Full Flow vs Inflow
plt.subplot(2, 1, 1)
sns.scatterplot(data=df, x='Inflow', y='Full_Flow', alpha=0.5)
plt.title('Full Flow vs Inflow\n(Points along diagonal line = Full Flow matches Inflow)', pad=20)
plt.xlabel('Inflow (cfs)')
plt.ylabel('Full Flow (cfs)')

# Add diagonal line for reference
max_val = max(df['Inflow'].max(), df['Full_Flow'].max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='1:1 Line\n(Full Flow = Inflow)')
plt.legend()

# Add text annotation explaining the plot
plt.text(0.02, 0.98, 
         'Points above line: Full Flow > Inflow\nPoints below line: Full Flow < Inflow', 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Time series plot showing correlation over time
plt.subplot(2, 1, 2)

# Calculate and plot monthly correlation
monthly_corr = df.groupby(df['Date'].dt.to_period('M')).apply(
    lambda x: x['Inflow'].corr(x['Full_Flow'])
).reset_index()
monthly_corr['Date'] = monthly_corr['Date'].dt.to_timestamp()

plt.plot(monthly_corr['Date'], monthly_corr[0], 
         label='Monthly Correlation', 
         color='blue', 
         alpha=0.7)

plt.title('Monthly Correlation between Inflow and Full Flow\n(Higher values = stronger relationship)', pad=20)
plt.xlabel('Date')
plt.ylabel('Correlation Coefficient\n(1 = perfect correlation, 0 = no correlation)')
plt.legend()
plt.grid(True)

# Add text annotation explaining correlation
plt.text(0.02, 0.98, 
         'Correlation = 1: Perfect match between Inflow and Full Flow\n' +
         'Correlation = 0: No relationship\n' +
         'Higher correlation suggests flow-through conditions',
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('full_flow_analysis.png')

# Print statistics
print("\nFlow Statistics:")
print(f"Average Inflow: {df['Inflow'].mean():.1f} cfs")
print(f"Average Full Flow: {df['Full_Flow'].mean():.1f} cfs")
print(f"Max Inflow: {df['Inflow'].max():.1f} cfs")
print(f"Max Full Flow: {df['Full_Flow'].max():.1f} cfs")

# Calculate overall correlation
overall_corr = df['Inflow'].corr(df['Full_Flow'])
print(f"\nOverall correlation: {overall_corr:.3f}")

# Calculate seasonal correlations
df['Month'] = df['Date'].dt.month
seasonal_corr = df.groupby('Month')[['Inflow', 'Full_Flow']].corr().unstack().iloc[:, 1]
print("\nMonthly correlations:")
for month in range(1, 13):
    print(f"Month {month}: {seasonal_corr[month]:.3f}")
