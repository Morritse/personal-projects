# Water Data Analysis Project

## Historical Weather Station Data Collection

This project includes analysis of water-related metrics including flow rates, precipitation, snow melt, and reservoir data. We are expanding the data collection to include historical weather station data from the California Data Exchange Center (CDEC).

### Data Structure
The historical data template includes:
- Date information (Date, WaterYear, DayOfYear, Month, Day)
- Flow metrics (TotalIn, Inflow, BaseFlow)
- Special flow calculations (KondolfFlow)
- Water management data (Diversion)

### CDEC Data Collection
The California Data Exchange Center (CDEC) provides access to real-time and historical water data. Key data points we aim to collect include:
- Precipitation
- Temperature
- Snow water content
- River stage/flow
- Reservoir storage

Data will be collected through CDEC's web services and organized to match our existing data structure where applicable.

### Project Organization
- `src/utils/` - Utility scripts including data collection tools
- `src/analysis/` - Analysis scripts organized by data type
- `output/` - Generated analysis results and plots
