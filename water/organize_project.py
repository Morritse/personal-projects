import os
import shutil

# Create main directory structure
directories = {
    'src': {
        'analysis': {
            'snow': {},  # Snow-related analysis
            'flow': {},  # Flow-related analysis
            'reservoir': {},  # Reservoir analysis
            'precipitation': {},  # Precipitation analysis
            'seasonal': {},  # Seasonal patterns
        },
        'utils': {},  # Utility scripts
    },
    'data': {
        'raw': {},  # Original data files
        'processed': {},  # Processed/intermediate data
    },
    'output': {
        'plots': {
            'snow': {},
            'flow': {},
            'reservoir': {},
            'precipitation': {},
            'seasonal': {},
        },
        'reports': {},
    },
    'archive': {},  # For deprecated files
}

# Create directory structure
for parent_dir, subdirs in directories.items():
    for subdir, subsubdirs in subdirs.items():
        path = os.path.join(parent_dir, subdir)
        os.makedirs(path, exist_ok=True)
        for subsubdir in subsubdirs:
            os.makedirs(os.path.join(path, subsubdir), exist_ok=True)

# File categorization
file_mapping = {
    # Analysis scripts
    'src/analysis/snow': [
        'analyze_snow_melt.py',
        'analyze_snow_flow.py',
        'analyze_snow_volume.py',
        'analyze_snow_melt_rate.py',
        'analyze_snow_flow_sequence.py',
    ],
    'src/analysis/flow': [
        'analyze_flow_volume.py',
        'analyze_full_flow.py',
    ],
    'src/analysis/precipitation': [
        'analyze_precip_inflow.py',
        'analyze_regional_precip.py',
    ],
    'src/analysis/reservoir': [
        'analyze_spills.py',
        'analyze_sjr_spills.py',
        'visualize_reservoir.py',
    ],
    'src/analysis/seasonal': [
        'analyze_patterns.py',
        'seasonal_patterns.py',
        'key_patterns.py',
    ],
    'src/utils': [
        'create_base_structure.py',
    ],
    
    # Data files
    'data/raw': [
        'fnf2010_2020.csv',
        'GRM.csv',
        'HNT.csv',
        'KSP.csv',
        'VLC.csv',
        'precipitation.csv',
        'inflow.csv',
        'friant_out.csv',
        'madera_out.csv',
        'required_out.csv',
        'actual_sjr_out.csv',
        'historical_reservoir_level.csv',
    ],
    'data/processed': [
        'water_years_base.csv',
    ],
    
    # Plot outputs
    'output/plots/snow': [
        'snow_melt_analysis.png',
        'snow_flow_analysis.png',
        'snow_flow_sequence.png',
        'snow_volume_analysis.png',
        'snow_melt_rate_analysis.png',
    ],
    'output/plots/flow': [
        'flow_volume_analysis.png',
        'full_flow_analysis.png',
    ],
    'output/plots/precipitation': [
        'regional_precip_analysis.png',
        'temp_inflow_analysis.png',
    ],
    'output/plots/reservoir': [
        'spill_analysis.png',
        'sjr_spill_analysis.png',
    ],
    'output/plots/seasonal': [
        'seasonal_patterns.png',
        'cross_correlation.png',
    ],
    
    # Archive deprecated files
    'archive': [
        'deprec/analyze_historical.py',
        'deprec/analyze_reservoir.py',
        'deprec/create_historical_df.py',
        'deprec/diversion.csv',
        'deprec/historical_analysis_1952.csv',
        'deprec/historical_analysis_1952.png',
        'deprec/historical_data_template.csv',
        'deprec/historical_data_template copy.csv',
        'deprec/inflow.csv',
        'deprec/kondolf.csv',
        'deprec/o1backup.py',
        'deprec/reservoir_analysis.csv',
        'deprec/reservoir_analysis.png',
        'deprec/water_model.csv',
    ],
}

# Move files to their new locations
for target_dir, files in file_mapping.items():
    os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists
    for file_path in files:
        if os.path.exists(file_path):
            target_path = os.path.join(target_dir, os.path.basename(file_path))
            try:
                print(f"Moving {file_path} -> {target_path}")
                shutil.move(file_path, target_path)
            except Exception as e:
                print(f"Error moving {file_path}: {str(e)}")

# Clean up empty data directories
for dir_to_remove in ['datacsv', 'data']:
    if os.path.exists(dir_to_remove):
        try:
            shutil.rmtree(dir_to_remove)
            print(f"Removed empty directory: {dir_to_remove}")
        except:
            print(f"Could not remove {dir_to_remove} (may not be empty)")

print("\nProject structure reorganized!")
print("\nNew structure:")
print("├── src/")
print("│   ├── analysis/")
print("│   │   ├── snow/        # Snow melt and flow analysis")
print("│   │   ├── flow/        # General flow analysis")
print("│   │   ├── reservoir/   # Reservoir operations")
print("│   │   ├── precipitation/# Precipitation analysis")
print("│   │   └── seasonal/    # Seasonal patterns")
print("│   └── utils/           # Utility scripts")
print("├── data/")
print("│   ├── raw/             # Original data files")
print("│   └── processed/       # Processed data")
print("├── output/")
print("│   ├── plots/           # Analysis plots by category")
print("│   └── reports/         # Analysis reports/summaries")
print("└── archive/             # Deprecated files")
