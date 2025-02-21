"""
Set up data directory structure and organize data files.

This script:
1. Creates necessary directories
2. Moves/copies data files to the right locations
"""

from pathlib import Path
import shutil

def setup_data_directories():
    """Create required directory structure."""
    # Base directories
    base_dir = Path('data')
    cdec_dir = base_dir / 'cdec'
    
    # Create directories
    for dir_path in [base_dir, cdec_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
        
def organize_data_files():
    """Move data files to appropriate locations."""
    # Source files in current directory
    source_files = {
        'ff.csv': 'comparative_sensor_8_full_natural_flow.csv',
        'historical_data_template.csv': 'historical_data_template.csv'
    }
    
    cdec_dir = Path('data/cdec')
    
    # Move/copy files
    for src_name, dest_name in source_files.items():
        src_path = Path(src_name)
        if src_path.exists():
            dest_path = cdec_dir / dest_name
            shutil.copy2(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
        else:
            print(f"Warning: Source file not found: {src_path}")

def main():
    """Set up data directory structure and organize files."""
    print("Setting up data directories...")
    setup_data_directories()
    
    print("\nOrganizing data files...")
    organize_data_files()
    
    print("\nSetup complete!")

if __name__ == "__main__":
    main()
