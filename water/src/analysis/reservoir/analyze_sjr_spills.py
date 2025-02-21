import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(20, 15))

# Read data
df = pd.read_csv('water_years_base.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Calculate excess flow
df['SJR_Excess'] = df['Actual_SJR_Out'] - df['Required_Out']

# Identify spill events and excess flow events
df['Is_Spilling'] = df['Reservoir_Level'] > 522500
df['Has_Excess_Flow'] = df['SJR_Excess'] > 0

# Create subplots
fig = plt.figure(figsize=(20, 15))

# 1. Time series of excess flow during spills
ax1 = plt.subplot(3, 1, 1)
spill_periods = df[df['Is_Spilling']]
ax1.plot(df['Date'], df['SJR_Excess'], color='blue', alpha=0.3, label='Normal Operation')
ax1.plot(spill_periods['Date'], spill_periods['SJR_Excess'], 
         color='red', alpha=0.7, label='During Spills')
ax1.set_title('SJR Excess Flow (Actual - Required) Over Time')
ax1.set_ylabel('Excess Flow')
ax1.legend()

# 2. Box plot of excess flow by water year type, split by spill/no-spill
ax2 = plt.subplot(3, 1, 2)
sns.boxplot(data=df, x='Water_Year_Type', y='SJR_Excess', hue='Is_Spilling')
ax2.set_title('Distribution of Excess Flow by Year Type and Spill Status')
ax2.set_ylabel('Excess Flow')

# 3. Scatter plot of reservoir level vs excess flow
ax3 = plt.subplot(3, 1, 3)
sns.scatterplot(data=df, x='Reservoir_Level', y='SJR_Excess', 
                hue='Water_Year_Type', alpha=0.5)
ax3.axvline(x=522500, color='red', linestyle='--', label='Spill Level')
ax3.set_title('Reservoir Level vs Excess Flow')
ax3.set_xlabel('Reservoir Level')
ax3.set_ylabel('Excess Flow')

plt.tight_layout()
plt.savefig('sjr_spill_analysis.png')

# Print statistics
print("\nSpill and Excess Flow Analysis:")

# Overall statistics
total_days = len(df)
spill_days = df['Is_Spilling'].sum()
excess_days = df['Has_Excess_Flow'].sum()
coincident_days = (df['Is_Spilling'] & df['Has_Excess_Flow']).sum()

print(f"\nOverall Statistics:")
print(f"Total days analyzed: {total_days}")
print(f"Days with spills: {spill_days} ({spill_days/total_days*100:.1f}%)")
print(f"Days with excess flow: {excess_days} ({excess_days/total_days*100:.1f}%)")
print(f"Days with both: {coincident_days} ({coincident_days/total_days*100:.1f}%)")

# Average excess flow during spills vs non-spills
print("\nAverage Excess Flow:")
print("During spills:", df[df['Is_Spilling']]['SJR_Excess'].mean())
print("During normal operation:", df[~df['Is_Spilling']]['SJR_Excess'].mean())

# Analysis by water year type
print("\nAnalysis by Water Year Type:")
for year_type in sorted(df['Water_Year_Type'].unique()):
    type_data = df[df['Water_Year_Type'] == year_type]
    type_spills = type_data['Is_Spilling'].sum()
    type_excess = type_data['Has_Excess_Flow'].sum()
    type_both = (type_data['Is_Spilling'] & type_data['Has_Excess_Flow']).sum()
    
    print(f"\nType {year_type}:")
    print(f"Spill days: {type_spills}")
    print(f"Excess flow days: {type_excess}")
    print(f"Days with both: {type_both}")
    if type_spills > 0:
        print(f"Average excess during spills: {type_data[type_data['Is_Spilling']]['SJR_Excess'].mean():.1f}")

# Correlation analysis
correlation = df['Reservoir_Level'].corr(df['SJR_Excess'])
print(f"\nCorrelation between Reservoir Level and Excess Flow: {correlation:.3f}")
