import sys
import os
import importlib.util
from datetime import datetime

# Add eod_scripts to Python path
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eod_scripts')
sys.path.append(SCRIPT_DIR)

def import_module(name, path):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def run_pipeline():
    print(f"\n=== Starting EOD Pipeline {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    try:
        # 1. Fetch EOD Data
        print("Step 1: Fetching EOD Data...")
        fetch_eod = import_module('fetch_eod_data', os.path.join(SCRIPT_DIR, 'fetch_eod_data.py'))
        fetch_eod.main()
        print("✓ EOD Data fetched successfully\n")
        
        # 2. Aggregate EOD Data
        print("Step 2: Aggregating EOD Data...")
        aggregate_eod = import_module('aggregate_eod_data', os.path.join(SCRIPT_DIR, 'aggregate_eod_data.py'))
        aggregate_eod.aggregate_eod_data()
        print("✓ EOD Data aggregated successfully\n")
        
        # 3. Generate EOD Signals
        print("Step 3: Generating EOD Signals...")
        end_of_day = import_module('end_of_day', os.path.join(SCRIPT_DIR, 'end_of_day.py'))
        end_of_day.main()
        print("✓ EOD Signals generated successfully\n")
        
        print("=== EOD Pipeline Completed Successfully ===")
        
    except Exception as e:
        print(f"\n!!! Error in EOD Pipeline !!!")
        print(f"Error details: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()
