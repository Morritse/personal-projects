import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------
# 1. Read Data
# -------------------------------------------
# Suppose your CSV file is named "reservoir_data.csv" with columns:
#   Date,WaterYear,DayOfYear,Month,Day,WaterYearType,TotalIn,Inflow,BaseFlow,KondolfFlow,Diversion
# Adjust the path and parse_dates if needed.

df = pd.read_csv("reservoir_data.csv",
                 parse_dates=["Date"])

# -------------------------------------------
# 2. Define Key Constants
# -------------------------------------------
RESERVOIR_CAPACITY = 525000  # AF

# -------------------------------------------
# 3. Initialize Calculation Columns
# -------------------------------------------
# We'll add columns to store daily reservoir volume ("CalcVolume") and daily spillage.

df["CalcVolume"] = 0.0
df["Spillage"] = 0.0

# -------------------------------------------
# 4. Process by Water Year
# -------------------------------------------
# Approach:
#   - Group by 'WaterYear'
#   - For each group, the FIRST day is 'TotalIn'
#   - For the subsequent days, compute new volume from old volume + inflow - outflows
#   - If it exceeds 525k => spillage
#   - Then store results in new columns

grouped = df.groupby("WaterYear")

all_rows = []  # We will store the updated daily data here

for water_year, group_df in grouped:
    # Sort each group by date just in case (or by DayOfYear if guaranteed sorted).
    group_df = group_df.sort_values("Date").copy()
    
    # The first day’s CalcVolume = the day’s "TotalIn"
    first_idx = group_df.index[0]
    group_df.loc[first_idx, "CalcVolume"] = group_df.loc[first_idx, "TotalIn"]
    group_df.loc[first_idx, "Spillage"]   = 0.0
    
    # Iterate over the rest of the group in chronological order
    prev_volume = group_df.loc[first_idx, "CalcVolume"]
    
    for idx in group_df.index[1:]:
        row = group_df.loc[idx]
        # daily net change
        daily_inflow   = row["Inflow"]
        daily_baseflow = row["BaseFlow"]
        daily_kflow    = row["KondolfFlow"]
        daily_div      = row["Diversion"]
        
        # compute new volume
        new_volume = prev_volume + daily_inflow - (daily_baseflow + daily_kflow + daily_div)
        
        # spillage check
        spillage = 0.0
        if new_volume > RESERVOIR_CAPACITY:
            spillage  = new_volume - RESERVOIR_CAPACITY
            new_volume = RESERVOIR_CAPACITY
        
        # assign back to dataframe
        group_df.loc[idx, "CalcVolume"] = new_volume
        group_df.loc[idx, "Spillage"]   = spillage
        
        # update prev_volume
        prev_volume = new_volume
    
    all_rows.append(group_df)

# Concatenate all the updated groups back
df_updated = pd.concat(all_rows).sort_values("Date")

# -------------------------------------------
# 5. Plot
# -------------------------------------------
# We can do a simple line plot of reservoir volume.
# And optionally a bar plot or line plot of daily spillage.

fig, ax1 = plt.subplots(figsize=(10,6))

ax1.plot(df_updated["Date"],
         df_updated["CalcVolume"],
         label="Calculated Reservoir Volume (AF)",
         color="C0")

# Show the 525k capacity as a horizontal line
ax1.axhline(RESERVOIR_CAPACITY, color="r", linestyle="--", label="525,000 AF Capacity")

ax1.set_xlabel("Date")
ax1.set_ylabel("Reservoir Volume (AF)")
ax1.set_title("Daily Reservoir Storage vs. Capacity")

# Optionally add spillage as a second y-axis
ax2 = ax1.twinx()
ax2.bar(df_updated["Date"],
        df_updated["Spillage"],
        width=1,
        alpha=0.2,
        color="C1",
        label="Daily Spillage (AF)")

ax2.set_ylabel("Daily Spillage (AF)")

# Put legends together
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.tight_layout()
plt.show()
