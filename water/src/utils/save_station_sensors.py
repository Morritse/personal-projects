"""
Save station sensor information to a structured JSON file.
"""

import json
from typing import Dict, List, Set
from pathlib import Path
from datetime import datetime

def create_station_sensor_dict() -> Dict:
    """Create a dictionary of stations and their available sensors."""
    return {
        'AGP': {
            'sensors': {
                '18': {'name': 'SNOW DEPTH', 'durations': ['daily', 'monthly']},
                '3': {'name': 'SNOW, WATER CONTENT', 'durations': ['daily', 'monthly']},
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '82': {'name': 'SNOW, WATER CONTENT(REVISED)', 'durations': ['daily', 'monthly']}
            }
        },
        'AGW': {
            'sensors': {
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']}
            }
        },
        'BDM': {
            'sensors': {
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']}
            }
        },
        'CNV': {
            'sensors': {
                '15': {'name': 'STORAGE', 'durations': ['daily']},
                '6': {'name': 'EVAPORATION', 'durations': ['daily']}
            }
        },
        'FLR': {
            'sensors': {
                '15': {'name': 'STORAGE', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']},
                '6': {'name': 'EVAPORATION', 'durations': ['daily']}
            }
        },
        'FRT': {
            'sensors': {
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']}
            }
        },
        'GRM': {
            'sensors': {
                '18': {'name': 'SNOW DEPTH', 'durations': ['daily']},
                '3': {'name': 'SNOW, WATER CONTENT', 'durations': ['daily']},
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']},
                '82': {'name': 'SNOW, WATER CONTENT(REVISED)', 'durations': ['daily']}
            }
        },
        'GTM': {
            'sensors': {
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']}
            }
        },
        'HNT': {
            'sensors': {
                '15': {'name': 'STORAGE', 'durations': ['daily']},
                '18': {'name': 'SNOW DEPTH', 'durations': ['daily']},
                '3': {'name': 'SNOW, WATER CONTENT', 'durations': ['daily']},
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']},
                '6': {'name': 'EVAPORATION', 'durations': ['daily']},
                '82': {'name': 'SNOW, WATER CONTENT(REVISED)', 'durations': ['daily']}
            }
        },
        'KKR': {
            'sensors': {
                '15': {'name': 'STORAGE', 'durations': ['daily']},
                '6': {'name': 'EVAPORATION', 'durations': ['daily']}
            }
        },
        'KSP': {
            'sensors': {
                '18': {'name': 'SNOW DEPTH', 'durations': ['daily']},
                '3': {'name': 'SNOW, WATER CONTENT', 'durations': ['daily']},
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']},
                '82': {'name': 'SNOW, WATER CONTENT(REVISED)', 'durations': ['daily']}
            }
        },
        'MIL': {
            'sensors': {
                '15': {'name': 'STORAGE', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']},
                '6': {'name': 'EVAPORATION', 'durations': ['daily']}
            }
        },
        'NFR': {
            'sensors': {
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']}
            }
        },
        'NTP': {
            'sensors': {
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']}
            }
        },
        'PSR': {
            'sensors': {
                '18': {'name': 'SNOW DEPTH', 'durations': ['daily']},
                '3': {'name': 'SNOW, WATER CONTENT', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']},
                '82': {'name': 'SNOW, WATER CONTENT(REVISED)', 'durations': ['daily']}
            }
        },
        'RDN': {
            'sensors': {
                '15': {'name': 'STORAGE', 'durations': ['daily']},
                '6': {'name': 'EVAPORATION', 'durations': ['daily']}
            }
        },
        'SBF': {
            'sensors': {
                '8': {'name': 'FULL NATURAL FLOW', 'durations': ['daily']}
            }
        },
        'TMR': {
            'sensors': {
                '18': {'name': 'SNOW DEPTH', 'durations': ['daily']},
                '3': {'name': 'SNOW, WATER CONTENT', 'durations': ['daily']},
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '45': {'name': 'PRECIPITATION', 'durations': ['daily']},
                '82': {'name': 'SNOW, WATER CONTENT(REVISED)', 'durations': ['daily']}
            }
        },
        'VLC': {
            'sensors': {
                '18': {'name': 'SNOW DEPTH', 'durations': ['daily', 'monthly']},
                '3': {'name': 'SNOW, WATER CONTENT', 'durations': ['daily', 'monthly']},
                '30': {'name': 'TEMPERATURE, AIR AVERAGE', 'durations': ['daily']},
                '31': {'name': 'TEMPERATURE, AIR MAXIMUM', 'durations': ['daily']},
                '32': {'name': 'TEMPERATURE, AIR MINIMUM', 'durations': ['daily']},
                '82': {'name': 'SNOW, WATER CONTENT(REVISED)', 'durations': ['daily', 'monthly']}
            }
        }
    }

def save_station_sensors(output_file: str = "data/cdec/station_sensors.json"):
    """Save the station sensor dictionary to a JSON file."""
    # Create the directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the station sensor dictionary
    station_sensors = create_station_sensor_dict()
    
    # Add metadata
    data = {
        'metadata': {
            'total_stations': len(station_sensors),
            'date_generated': datetime.now().isoformat(),
            'sensor_types': sorted(list(set(
                sensor['name']
                for station in station_sensors.values()
                for sensor in station['sensors'].values()
            ))),
            'duration_types': sorted(list(set(
                duration
                for station in station_sensors.values()
                for sensor in station['sensors'].values()
                for duration in sensor['durations']
            )))
        },
        'stations': station_sensors
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Station sensor information saved to {output_path}")
    print("\nMetadata Summary:")
    print(f"Total stations: {data['metadata']['total_stations']}")
    print("\nAvailable sensor types:")
    for sensor in data['metadata']['sensor_types']:
        print(f"  - {sensor}")
    print("\nAvailable duration types:")
    for duration in data['metadata']['duration_types']:
        print(f"  - {duration}")

if __name__ == "__main__":
    save_station_sensors()
