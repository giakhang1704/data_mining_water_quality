# Water Quality Data Mining Project

This project analyzes water quality data from marine and inland monitoring stations. It builds predictive models, visualizes the data, and performs clustering to assess environmental quality.

## Main components
- `DATA_MINING_OCEAN.ipynb`: notebook for processing and analyzing marine water quality data.
- `DATA_MINIG_LAND.ipynb`: notebook for processing and analyzing inland water quality data.
- `streamlit_app.py`: interactive dashboard application for viewing results and visualizations.
- `visualize.html`: static HTML page displaying water quality trends for individual stations over time, based on clustering results.
- `dataset/`: contains raw data files and processed output.
- `README.md`: usage guide and project overview.

## Data
- `dataset/monthly_ocean.csv`: monthly marine water quality data.
- `dataset/full_dataset.csv`, `dataset/weekly_land.csv`, `dataset/daily_land.csv`: inland water quality data at different time resolutions.
- `dataset/OceanQuality_Clustered_Results.csv`: marine water quality clustering results.
- `dataset/WaterQuality_Clustered_Results.csv`: inland water quality clustering results.

## How to use
1. Activate the Python environment:
```bash
# Windows PowerShell
& .\myenv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

## Project goals
- Preprocess and clean water quality data.
- Explore features and trends over time.
- Build predictive models and fill missing values.
- Perform clustering to classify monitoring stations by water quality.
- Visualize the results using charts, maps, and a dashboard.
