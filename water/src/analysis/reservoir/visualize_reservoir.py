import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
sns.set_palette("husl")

# Read the data
df = pd.read_csv('water_years_base.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create the plot
plt.figure(figsize=(20, 10))

# Plot reservoir level
sns.lineplot(data=df, x='Date', y='Reservoir_Level', linewidth=1)

# Add water year boundaries and annotations
water_years = df['Water_Year'].unique()
for year in water_years:
    # Get first date of each water year
    year_start = df[df['Water_Year'] == year]['Date'].iloc[0]
    year_type = df[df['Water_Year'] == year]['Water_Year_Type'].iloc[0]
    
    # Add vertical line for water year boundary
    plt.axvline(x=year_start, color='gray', linestyle='--', alpha=0.3)
    
    # Add water year and type annotation at top
    plt.text(year_start, plt.ylim()[1], f'{year}\nType {year_type}', 
             rotation=45, ha='right', va='top')

# Add horizontal line for spill level
plt.axhline(y=522500, color='red', linestyle='--', alpha=0.5, label='Spill Level (522,500)')

plt.title('Reservoir Level Over Time with Water Year Boundaries')
plt.xlabel('Date')
plt.ylabel('Reservoir Level')
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('reservoir_level_timeline.png')
