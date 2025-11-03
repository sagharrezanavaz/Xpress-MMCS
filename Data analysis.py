#%%
from pathlib import Path
import re
import pandas as pd
import sys
#%%

# ----- USER CONFIG -----
# Directory where your 36 CSV files live
INPUT_DIR = Path(r'E:\Uni_PGT\counts-data')  # <- updated path
# Output filename for the combined CSV
OUTPUT_FILE = Path(r'E:\Uni_PGT\counts-data\combined_od_with_datew.csv')
# Pattern to match files (will match filenames containing YYYY_MM or YYYY-MM)
GLOB_PATTERN = '*_counts*.csv'
# ------------------------


def extract_year_month_from_name(fname: str):
    
    """Return (year, month) tuple if found in filename, else None."""
    # look for 4-digit year, separator (_ or -), 2-digit month
    m = re.search(r'(\d{4})[_-](\d{2})', fname)
    if not m:
        return None
    year, month = m.group(1), m.group(2)
    return year, month


def main(input_dir: Path, output_file: Path):
    files = sorted(input_dir.glob(GLOB_PATTERN))
    if not files:
        print(f'No files found matching pattern {GLOB_PATTERN} in {input_dir.resolve()}')
        sys.exit(1)

    dfs = []
    skipped = []

    for f in files:
        ym = extract_year_month_from_name(f.name)
        if ym is None:
            skipped.append(f.name)
            continue

        year, month = ym
        # read CSV
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f'Failed to read {f.name}: {e}')
            skipped.append(f.name)
            continue

        # add date columns in two common formats:
        # 'year_month' = 'YYYY-MM' and 'month_year' = 'MM/YYYY' (user asked for m/y)
        df['year_month'] = f"{year}-{month}"
        df['month_year'] = f"{month}/{year}"

        dfs.append(df)

    if not dfs:
        print('No CSVs successfully read (maybe filename pattern is different). Files skipped:')
        print('\n'.join(skipped))
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    # Optional: reorder so date columns are near the front
    cols = list(combined.columns)
    for col in ['year_month', 'month_year']:
        if col in cols:
            cols.insert(0, cols.pop(cols.index(col)))
    combined = combined[cols]

    # Save the combined CSV
    combined.to_csv(output_file, index=False)
    print(f'Combined {len(dfs)} files into {output_file} (total rows: {len(combined)})')
    if skipped:
        print('Skipped files (no YYYY_MM found or read error):')
        print('\n'.join(skipped))


if __name__ == '__main__':
    main(INPUT_DIR, OUTPUT_FILE)

#%%
df=pd.read_csv(r'E:\Uni_PGT\counts-data\combined_od_with_date.csv')
df.info()
df.describe()
#%%
REPORT_FILE = Path(r'E:\Uni_PGT\counts-data\data_quality_report.txt')
# Ensure correct data types
df['start_station_id'] = pd.to_numeric(df['start_station_id'], errors='coerce').astype('Int64')
df['end_station_id'] = pd.to_numeric(df['end_station_id'], errors='coerce').astype('Int64')
df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
df['trip_count'] = pd.to_numeric(df['trip_count'], errors='coerce').astype('Int64')

# ----- MISSING VALUES -----
missing_summary = df.isna().sum()
missing_rows = df[df.isna().any(axis=1)]

if not missing_rows.empty:
    print(f"Found {len(missing_rows):,} rows with missing values — will drop them.")
    df = df.dropna()
#%%

# ----- DUPLICATES -----
num_dupes = df.duplicated().sum()
if num_dupes > 0:
    print(f"Dropping {num_dupes:,} duplicate rows.")
    df = df.drop_duplicates()
#%%

# ----- VALUE VALIDATION -----
# Check valid range for hour (0–23 expected)
invalid_hours = df[~df['hour'].between(0, 24)]
if not invalid_hours.empty:
    print(f"Found {len(invalid_hours):,} rows with invalid hour values (outside 0–24). Fixing...")
    df = df[df['hour'].between(0, 24)]
#%%
# Check station ID ranges
df_s=pd.read_csv(r'E:\Uni_PGT\station_data.csv')
m=list(df_s['station_id'].unique())
w=df['start_station_id'].unique()
missing_ids = [s for s in w if s not in m]
print(missing_ids)
print(len(missing_ids))

#%%

# ----- REPORT -----
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write('OD Matrix Data Quality Report\n')
    f.write('=' * 40 + '\n\n')
    f.write(f'Total rows after cleaning: {len(df):,}\n')
    f.write(f'Duplicates removed: {num_dupes}\n')
    f.write(f'Missing rows removed: {len(missing_rows)}\n')
    f.write(f'Invalid hour rows removed: {len(invalid_hours)}\n\n')


    f.write('Trip count summary (post-clean):\n')
    f.write(str(df['trip_count'].describe()) + '\n\n')


print(f'Data quality report saved to: {REPORT_FILE}')


#%%
"""
generate_od_heatmaps.py

Generates OD heatmap images for each month (36 files) plus one combined OD heatmap
from the combined OD CSV created earlier.

Notes / behavior:
 - The script reads 'combined_od_with_date.csv' and expects columns:
   ['year_month','month_year','start_station_id','end_station_id','hour','trip_count']
 - It will order stations by numeric station id (ascending). If you have a station
   reference file and want a specific ordering, set STATION_REF_FILE.
 - For visualization, the script uses np.log1p on counts to reduce skew. The saved
   CSVs keep raw aggregated counts.

Run:
    python generate_od_heatmaps.py

"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ----- USER CONFIG -----
OUTPUT_DIR = Path(r'E:\Uni_PGT\counts-data\heatmaps')
STATION_REF_FILE = Path(r'E:\Uni_PGT\station_data.csv')  # set to None to order by numeric id
LOG_DISPLAY = True   # use log1p for display to reduce skew
DPI = 150
# ------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# If no station reference, build station list from data
all_stations = np.union1d(df['start_station_id'].unique(), df['end_station_id'].unique()).astype(int)
all_stations_sorted = sorted([int(x) for x in all_stations])
n_stations = len(all_stations_sorted)
stations = sorted([int(x) for x in all_stations])
print(f'Total stations used for matrices: {n_stations}')

# helper to make pivot and plot

def make_pivot(df_subset, stations, aggcol='trip_count'):
    # aggregate counts to ensure one cell per pair
    agg = df_subset.groupby(['start_station_id', 'end_station_id'])[aggcol].sum().reset_index()
    pivot = agg.pivot(index='start_station_id', columns='end_station_id', values=aggcol).reindex(index=stations, columns=stations).fillna(0)
    return pivot


def plot_matrix(matrix_df, title, outpath, log_display=LOG_DISPLAY, dpi=DPI):
    arr = matrix_df.values.astype(float)
    display_arr = np.log1p(arr) if log_display else arr

    # dynamic figsize: cap sizes to avoid enormous images
    height, width = arr.shape
    figsize = (min(20, max(6, width/10)), min(20, max(6, height/10)))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(display_arr, aspect='auto')

    ax.set_title(title)
    ax.set_xlabel('end_station_id (ordered)')
    ax.set_ylabel('start_station_id (ordered)')

    # reduce tick labels for readability: show first, middle, last
    if width <= 30:
        xticks = range(width)
        xtick_labels = [str(int(v)) for v in matrix_df.columns]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
    else:
        ax.set_xticks([0, width//2, width-1])
        ax.set_xticklabels([str(int(matrix_df.columns[0])), str(int(matrix_df.columns[width//2])), str(int(matrix_df.columns[-1]))], fontsize=8)

    if height <= 30:
        yticks = range(height)
        ytick_labels = [str(int(v)) for v in matrix_df.index]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, fontsize=6)
    else:
        ax.set_yticks([0, height//2, height-1])
        ax.set_yticklabels([str(int(matrix_df.index[0])), str(int(matrix_df.index[height//2])), str(int(matrix_df.index[-1]))], fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log1p(trip_count)' if log_display else 'trip_count')

    plt.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print(f'Saved heatmap: {outpath}')


# 1) Per-month heatmaps
unique_months = sorted(df['year_month'].unique())
print(f'Found {len(unique_months)} unique months (expected 36): {unique_months}')

for ym in unique_months:
    df_month = df[df['year_month'] == ym]
    pivot = make_pivot(df_month, stations)
    outpath = OUTPUT_DIR / f'heatmap_{ym}.png'
    title = f'OD heatmap {ym} (log display={LOG_DISPLAY})'
    plot_matrix(pivot, title, outpath)

# 2) Combined heatmap for all data
pivot_combined = make_pivot(df, stations)
# save numeric combined matrix
pivot_combined.to_csv(OUTPUT_DIR / 'od_matrix_combined.csv')
plot_matrix(pivot_combined, 'OD heatmap combined (log display={})'.format(LOG_DISPLAY), OUTPUT_DIR / 'heatmap_combined.png')

print('All done. Generated per-month and combined heatmaps in:')
print(OUTPUT_DIR.resolve())
#%%
"""
seasonal_hourly_heatmaps.py

Generates seasonal-hourly OD trip heatmaps:
 - One heatmap per year (season x hour matrix) for each year in the data
 - One combined heatmap over all years
 - Saves matrices as CSV and images to OUTPUT_DIR

Seasons used (Northern Hemisphere standard):
 - Winter: Dec, Jan, Feb (DJF)
 - Spring: Mar, Apr, May (MAM)
 - Summer: Jun, Jul, Aug (JJA)
 - Autumn: Sep, Oct, Nov (SON)

Expect input columns: ['year_month','month_year','start_station_id','end_station_id','hour','trip_count']

Run:
    python seasonal_hourly_heatmaps.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- USER CONFIG -----
OUTPUT_DIR = Path(r'E:\Uni_PGT\counts-data\heatmaps_seasonal')
DPI = 150
LOG_DISPLAY = False  # For seasonal-hour heatmaps we keep linear counts by default
# ------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_year_month(ym):
    if isinstance(ym, str):
        if '-' in ym:
            parts = ym.split('-')
            return int(parts[0]), int(parts[1])
        if '/' in ym:
            parts = ym.split('/')
            # assume MM/YYYY -> return (YYYY, MM)
            return int(parts[1]), int(parts[0])
    raise ValueError(f'Unrecognized year_month format: {ym}')

parsed = df['year_month'].apply(parse_year_month)
df['year'] = parsed.apply(lambda x: x[0])
df['month'] = parsed.apply(lambda x: x[1])

# season mapping
season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}

df['season'] = df['month'].map(season_map)

# Helper: build season-hour pivot for a given dataframe

def season_hour_pivot(df_subset):
    # aggregate trip counts by season and hour
    agg = df_subset.groupby(['season', 'hour'])['trip_count'].sum().reset_index()
    # ensure all seasons and hours 0-23 present
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    hours = list(range(24))
    pivot = agg.pivot(index='season', columns='hour', values='trip_count').reindex(index=seasons, columns=hours).fillna(0)
    return pivot

# plotting helper

def plot_season_hour(matrix_df, title, outpath, log_display=LOG_DISPLAY, dpi=DPI):
    arr = matrix_df.values.astype(float)
    display_arr = np.log1p(arr) if log_display else arr

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(display_arr, aspect='auto')

    ax.set_title(title)
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Season')

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([str(h) for h in range(0, 24, 2)])

    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log1p(trip_count)' if log_display else 'trip_count')

    plt.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print(f'Saved: {outpath}')

# 1) Per-year seasonal-hour heatmaps
years = sorted(df['year'].unique())
print(f'Found years: {years}')

for y in years:
    df_year = df[df['year'] == y]
    pivot = season_hour_pivot(df_year)
    csv_out = OUTPUT_DIR / f'season_hour_matrix_{y}.csv'
    img_out = OUTPUT_DIR / f'season_hour_heatmap_{y}.png'
    pivot.to_csv(csv_out)
    plot_season_hour(pivot, f'Season vs Hour - {y}', img_out)

# 2) Combined (all years)
pivot_all = season_hour_pivot(df)
pivot_all.to_csv(OUTPUT_DIR / 'season_hour_matrix_all_years.csv')
plot_season_hour(pivot_all, 'Season vs Hour - All years', OUTPUT_DIR / 'season_hour_heatmap_all_years.png')

print('Done. Results saved to:')
print(OUTPUT_DIR.resolve())

#%%
"""
combine_36_csvs_no_date.py

Reads all CSV files from cyclehire-cleandata named like 2018_10.csv ... 2021_09.csv
(or containing that yyyy_mm pattern in the filename), concatenates them into a single CSV
and saves it to cyclehire-cleandata\combined_all_periods.csv

Notes:
 - This script does NOT add a date column (as requested).
 - It will align columns by name; missing columns in some files will be filled with NaN.
 - It prints a short summary of files read and total rows combined.

Usage:
    python combine_36_csvs_no_date.py
"""

from pathlib import Path
import pandas as pd
import re

# ----- USER CONFIG -----
INPUT_DIR = Path(r'E:\Uni_PGT\cyclehire-cleandata')
OUTPUT_FILE = INPUT_DIR / 'combined_all_periods.csv'
GLOB_PATTERN = '*_*.csv'  # matches files with yyyy_mm in name like 2018_10.csv
FILE_FILTER_REGEX = re.compile(r'(20\d{2})[_-](0[1-9]|1[0-2])')  # restrict to yyyy_mm patterns
# ------------------------

# collect candidate files
files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
selected_files = [f for f in files if FILE_FILTER_REGEX.search(f.name)]

if not selected_files:
    raise FileNotFoundError(f'No files matching yyyy_mm pattern found in {INPUT_DIR}')

print(f'Found {len(selected_files)} files to combine:')
for f in selected_files:
    print(' -', f.name)

# read and concatenate
dfs = []
for f in selected_files:
    try:
        df = pd.read_csv(f)
        df['__source_file'] = f.name  # optional: keep which file the row came from
        dfs.append(df)
    except Exception as e:
        print(f'Warning: failed to read {f.name}: {e}')

if not dfs:
    raise ValueError('No files were successfully read.')

combined = pd.concat(dfs, ignore_index=True, sort=False)
combined.to_csv(OUTPUT_FILE, index=False)

print(f'Combined {len(dfs)} files into {OUTPUT_FILE} (total rows: {len(combined):,})')

# optional quick sanity print
print('\nColumn summary (name : non-null count):')
print(combined.notna().sum().sort_values(ascending=False).head(50))

print('\nDone.')

#%%
print(combined.info())
print(combined.describe())
#%%

import seaborn as sns

# Define the path where your data is stored
data_path = r'E:\Uni_PGT\cyclehire-cleandata'

# Convert started_at column to datetime
combined['started_at'] = pd.to_datetime(combined['started_at'], errors='coerce')

# Extract weekday and hour
combined['weekday'] = combined['started_at'].dt.day_name()
combined['hour'] = combined['started_at'].dt.hour

# Assign seasons based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

combined['season'] = combined['started_at'].dt.month.apply(get_season)

# Order weekdays for consistent plotting
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create output folder for heatmaps
output_dir = os.path.join(data_path, 'heatmaps_weekday_hour')
os.makedirs(output_dir, exist_ok=True)

# Function to plot and save heatmap
def plot_heatmap(data, title, filename):
    pivot_table = data.pivot_table(index='weekday', columns='hour', values='duration', aggfunc='count').fillna(0)
    pivot_table = pivot_table.reindex(weekday_order)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, cmap='YlGnBu')
    plt.title(title)
    plt.ylabel('Weekday')
    plt.xlabel('Hour of Day')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Generate heatmap for each season
for season, data in combined.groupby('season'):
    plot_heatmap(data, f'Trip Count by Hour and Weekday - {season}', f'heatmap_{season}.png')

# Generate heatmap for all data combined
plot_heatmap(combined, 'Trip Count by Hour and Weekday - All Data', 'heatmap_all_data.png')

print(f"Heatmaps saved in: {output_dir}")
#%%
"""
duration_and_od_analysis.py

Produces:
 - Histogram of trip durations (linear and log-scaled)
 - CSV and plot for average duration by hour of day
 - CSV and plot for average duration by weekday
 - OD heatmap (average duration per start_station_id x end_station_id)

Assumptions:
 - Combined CSV of the 36 files exists at: cyclehire-cleandata\combined_all_periods.csv
 - Columns include: started_at, ended_at, duration, start_station_id, end_station_id

Outputs are written to cyclehire-cleandata\analysis_outputs

Run:
    python duration_and_od_analysis.py
"""

# ----- USER CONFIG -----
OUTPUT_DIR = Path(r'E:\Uni_PGT\cyclehire-cleandata\analysis_outputs')
DPI = 150
# ------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data

# Convert started_at to datetime (coerce errors)
combined['started_at'] = pd.to_datetime(combined['started_at'], errors='coerce')
# Filter out rows with missing datetime or non-positive duration
initial_rows = len(combined)
combined = combined[combined['duration'].notna()]
combined = combined[combined['duration'] > 0]
combined = combined[combined['started_at'].notna()]
print(f'Kept {len(combined):,} rows (removed {initial_rows - len(combined):,} invalid rows)')

# --- 1) Histograms of duration ---
# Linear histogram
plt.figure(figsize=(8,4))
plt.hist(combined['duration'], bins=100, range=(0, combined['duration'].quantile(0.99)))
plt.title('Trip duration distribution (0-99th percentile)')
plt.xlabel('Duration')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'duration_histogram_linear.png', dpi=DPI)
plt.close()

# Log-scaled histogram (log1p)
plt.figure(figsize=(8,4))
plt.hist(np.log1p(combined['duration']), bins=100)
plt.title('Trip duration distribution (log1p)')
plt.xlabel('log1p(Duration)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'duration_histogram_log.png', dpi=DPI)
plt.close()

# Save basic stats
desc = combined['duration'].describe()
desc.to_csv(OUTPUT_DIR / 'duration_summary_stats.csv')

# --- 2) Average duration by hour of day ---
combined['hour'] = combined['started_at'].dt.hour
avg_by_hour = combined.groupby('hour')['duration'].mean().reindex(range(24)).fillna(0)
avg_by_hour.to_csv(OUTPUT_DIR / 'avg_duration_by_hour.csv')

plt.figure(figsize=(10,4))
plt.plot(avg_by_hour.index, avg_by_hour.values, marker='o')
plt.grid(True)
plt.xlabel('Hour of day')
plt.ylabel('Average duration')
plt.title('Average trip duration by hour of day')
plt.xticks(range(0,24))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'avg_duration_by_hour.png', dpi=DPI)
plt.close()

# --- 3) Average duration by weekday ---
combined['weekday'] = combined['started_at'].dt.day_name()
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
avg_by_weekday = combined.groupby('weekday')['duration'].mean().reindex(weekday_order)
avg_by_weekday.to_csv(OUTPUT_DIR / 'avg_duration_by_weekday.csv')

plt.figure(figsize=(8,4))
plt.bar(avg_by_weekday.index, avg_by_weekday.values)
plt.xlabel('Weekday')
plt.ylabel('Average duration (seconds)')
plt.title('Average trip duration by weekday')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'avg_duration_by_weekday.png', dpi=DPI)
plt.close()

# --- 4) OD heatmap: average duration per (start_station_id x end_station_id) ---
# To limit memory use, we will restrict to top N stations by activity, but also
# save a CSV of aggregated averages for all pairs.

# Aggregate per pair
pair_agg = combined.groupby(['start_station_id','end_station_id'])['duration'].agg(['mean','count']).reset_index()
pair_agg.rename(columns={'mean':'avg_duration','count':'trip_count'}, inplace=True)
pair_agg.to_csv(OUTPUT_DIR / 'od_pair_avg_duration_all_pairs.csv', index=False)

# Choose top stations by total trips (to make a manageable heatmap)
station_activity = pd.concat([combined['start_station_id'], combined['end_station_id']]).value_counts()
top_n = 100  # adjust if you want larger/smaller matrix
top_stations = station_activity.index[:top_n].astype(int).tolist()
print(f'Creating OD heatmap for top {len(top_stations)} stations by activity')

# Pivot for top stations
subset = pair_agg[pair_agg['start_station_id'].isin(top_stations) & pair_agg['end_station_id'].isin(top_stations)]
heat = subset.pivot(index='start_station_id', columns='end_station_id', values='avg_duration').reindex(index=top_stations, columns=top_stations).fillna(0)

# Plot heatmap (use log scale for color or linear depending on spread)
plt.figure(figsize=(12,10))
plt.imshow(np.log1p(heat.values), aspect='auto')
plt.colorbar(label='log1p(avg_duration)')
plt.title(f'OD average duration heatmap (top {len(top_stations)} stations)')
plt.xlabel('end_station_id (ordered by activity)')
plt.ylabel('start_station_id (ordered by activity)')
# keep tick labels sparse for readability
n = len(top_stations)
plt.xticks([0, n//2, n-1], [str(top_stations[0]), str(top_stations[n//2]), str(top_stations[-1])])
plt.yticks([0, n//2, n-1], [str(top_stations[0]), str(top_stations[n//2]), str(top_stations[-1])])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'od_avg_duration_heatmap_top{top_n}.png', dpi=DPI)
plt.close()

print('Analysis complete. Outputs saved to:', OUTPUT_DIR.resolve())

#%%
"""
save_hourly_maps_with_basemap.py

Saves per-hour origin-demand maps to disk (one PNG per hour) and also saves
weekday-hour heatmaps. Uses the in-memory `combined` DataFrame if present; otherwise
expects you to load it prior to running this script.

Outputs are written to: E:/Uni_PGT/visualisation_outputs/station_hour_maps

Requirements:
 - geopandas, matplotlib, seaborn, contextily (optional, for web basemap tiles)
   Install with: pip install geopandas matplotlib seaborn contextily

Notes on basemap: contextily fetches tiles from the web. If you have no internet,
this script will fall back to plotting station points on plain axes.

Run this in the same Python session where `combined` exists (not by re-loading the CSV).
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# ----- USER CONFIG -----
OUT_DIR = Path(r'E:/Uni_PGT/visualisation_outputs/station_hour_maps')
HEATMAP_DIR = Path(r'E:/Uni_PGT/visualisation_outputs/heatmaps_weekday_hour')
OUT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
DPI = 200
POINT_SCALE = 2000  # adjust to scale marker sizes (increase for larger markers)
USE_CONTEXTILY = True  # set False if you don't want to fetch basemap tiles
TILE_SOURCE = None  # default contextily source (None uses provider's default)
# ------------------------

print('Using in-memory DataFrame `combined`')
try:
    combined  # must exist in the environment
except NameError:
    raise RuntimeError('DataFrame `combined` not found in memory. Load it first.')

# Ensure datetime
combined['started_at'] = pd.to_datetime(combined['started_at'], errors='coerce')
combined = combined.dropna(subset=['started_at'])
combined['hour'] = combined['started_at'].dt.hour
combined['weekday'] = combined['started_at'].dt.day_name()

# Prepare station aggregated counts by hour
agg = combined.groupby(['start_station_id','start_station_latitude','start_station_longitude','hour']).size().reset_index(name='starts')
# If lat/lon columns have different names, try alternatives
if agg['start_station_latitude'].isna().all() or agg['start_station_longitude'].isna().all():
    # try alternative names in combined
    lat_col = None
    lon_col = None
    for c in combined.columns:
        if c.lower().endswith('latitude') and 'start' in c.lower():
            lat_col = c
        if c.lower().endswith('longitude') and 'start' in c.lower():
            lon_col = c
    if lat_col and lon_col:
        agg = combined.groupby(['start_station_id', lat_col, lon_col, 'hour']).size().reset_index(name='starts')
        agg = agg.rename(columns={lat_col:'start_station_latitude', lon_col:'start_station_longitude'})

# drop rows without coordinates
agg = agg.dropna(subset=['start_station_latitude','start_station_longitude'])

# Build GeoDataFrame (EPSG:4326)
agg['geometry'] = [Point(xy) for xy in zip(agg['start_station_longitude'].astype(float), agg['start_station_latitude'].astype(float))]
gdf = gpd.GeoDataFrame(agg, geometry='geometry', crs='EPSG:4326')

# Project to Web Mercator for contextily (if using basemap)
if USE_CONTEXTILY:
    try:
        gdf_web = gdf.to_crs(epsg=3857)
    except Exception as e:
        print('Could not reproject to WebMercator, disabling contextily basemap:', e)
        USE_CONTEXTILY = False
        gdf_web = gdf
else:
    gdf_web = gdf

# Determine map extent (in web mercator if using contextily)
minx, miny, maxx, maxy = gdf_web.total_bounds
xpad = (maxx - minx) * 0.08 if maxx > minx else 100
ypad = (maxy - miny) * 0.08 if maxy > miny else 100
extent = (minx - xpad, maxx + xpad, miny - ypad, maxy + ypad)

# Save one PNG per hour
print('Saving hourly station-origin maps to:', OUT_DIR)
for h in range(24):
    hour_gdf = gdf_web[gdf_web['hour'] == h]

    fig, ax = plt.subplots(figsize=(8, 8))

    # plot all stations faintly as background
    gdf_web.plot(ax=ax, color='lightgrey', markersize=5, alpha=0.5)

    if not hour_gdf.empty:
        # marker size scaled by sqrt to reduce dynamic range
        sizes = np.sqrt(hour_gdf['starts'].astype(float) + 1) * (POINT_SCALE / max(1, np.sqrt(hour_gdf['starts'].max()*200+ 1)))
        hour_gdf.plot(ax=ax, markersize=sizes, column='starts', cmap='Reds', legend=True, alpha=0.9)

    if USE_CONTEXTILY:
        try:
            import contextily as ctx
            if TILE_SOURCE is None:
                ctx.add_basemap(ax, crs=gdf_web.crs.to_string())
            else:
                ctx.add_basemap(ax, source=TILE_SOURCE, crs=gdf_web.crs.to_string())
        except Exception as e:
            print('contextily failed; continuing without basemap:', e)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()
    ax.set_title(f'Origin starts — hour {h}')

    outpath = OUT_DIR / f'station_starts_hour_{h:02d}.png'
    fig.savefig(outpath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

print('Hourly maps saved.')

# --- Save weekday-hour heatmap (counts) ---
print('Saving weekday-hour heatmap...')
combined['weekday'] = pd.Categorical(combined['weekday'], categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], ordered=True)
pivot = combined.pivot_table(index='weekday', columns=combined['hour'], values='start_station_id', aggfunc='count').fillna(0).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap='YlOrRd')
plt.title('Trip starts by weekday and hour')
plt.xlabel('Hour')
plt.ylabel('Weekday')
hm_path = HEATMAP_DIR / 'weekday_hour_heatmap_all_data.png'
plt.tight_layout()
plt.savefig(hm_path, dpi=DPI)
plt.close()
print('Weekday-hour heatmap saved to', hm_path)

print('\nAll outputs written to:', OUT_DIR.parent)

#%%
# hourly_maps_and_separate_bars_fixed_extent.py
#
# Creates per-hour maps (origins left, destinations right) and saves them to disk
# (maps and separate bar charts). This version computes a robust map extent using
# percentiles (2nd-98th) to ignore coordinate outliers so plots zoom into Edinburgh.
#
# Run in the same session where `combined` exists (the combined DataFrame you made).
# Requires: geopandas, matplotlib, seaborn, contextily (optional for basemap tiles).
#
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from matplotlib import cm
from matplotlib.colors import Normalize

# ---------- USER CONFIG ----------
OUT_DIR = Path(r'E:/Uni_PGT/visualisation_outputs/hourly_maps_v3_fixed_extent')
MAP_DIR = OUT_DIR / 'maps'
BARS_DIR = OUT_DIR / 'bars'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)
BARS_DIR.mkdir(parents=True, exist_ok=True)
DPI = 180
USE_CONTEXTILY = True     # set False if no internet or you don't want basemap tiles
POINT_BASE = 35           # base marker size (smaller => less overlap)
TOP_K_BARS = 12
CMAP_ORIG = 'Reds'
CMAP_DEST = 'Blues'
# ---------------------------------

# ensure combined exists
try:
    combined
except NameError:
    raise RuntimeError("DataFrame `combined` not found in memory. Load it first.")

# parse datetimes and hour
combined['started_at'] = pd.to_datetime(combined['started_at'], errors='coerce')
combined = combined.dropna(subset=['started_at'])
combined['hour'] = combined['started_at'].dt.hour

# flexible column detection
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

s_lat = find_col(combined, ['start_station_latitude','start_latitude','start_lat','start_station_lat'])
s_lon = find_col(combined, ['start_station_longitude','start_longitude','start_lon','start_station_lon'])
e_lat = find_col(combined, ['end_station_latitude','end_latitude','end_lat','end_station_lat'])
e_lon = find_col(combined, ['end_station_longitude','end_longitude','end_lon','end_station_lon'])

if not all([s_lat, s_lon, e_lat, e_lon]):
    raise RuntimeError('Could not find necessary start/end latitude/longitude columns in combined dataframe.')

# aggregate
orig_agg = combined.groupby(['start_station_id', s_lat, s_lon, 'hour']).size().reset_index(name='count')
orig_agg = orig_agg.rename(columns={s_lat:'lat', s_lon:'lon', 'start_station_id':'station_id'})

dest_agg = combined.groupby(['end_station_id', e_lat, e_lon, 'hour']).size().reset_index(name='count')
dest_agg = dest_agg.rename(columns={e_lat:'lat', e_lon:'lon', 'end_station_id':'station_id'})

orig_agg = orig_agg.dropna(subset=['lat','lon'])
dest_agg = dest_agg.dropna(subset=['lat','lon'])

# GeoDataFrames (WGS84)
gdf_o = gpd.GeoDataFrame(orig_agg, geometry=[Point(xy) for xy in zip(orig_agg['lon'].astype(float), orig_agg['lat'].astype(float))], crs='EPSG:4326')
gdf_d = gpd.GeoDataFrame(dest_agg, geometry=[Point(xy) for xy in zip(dest_agg['lon'].astype(float), dest_agg['lat'].astype(float))], crs='EPSG:4326')

# project to WebMercator for basemap if desired
use_ctx = False
if USE_CONTEXTILY:
    try:
        gdf_o_web = gdf_o.to_crs(epsg=3857)
        gdf_d_web = gdf_d.to_crs(epsg=3857)
        use_ctx = True
    except Exception as e:
        print('Contextily disabled (reprojection failed):', e)
        gdf_o_web = gdf_o
        gdf_d_web = gdf_d
else:
    gdf_o_web = gdf_o
    gdf_d_web = gdf_d

# ---------------------------
# ROBUST EXTENT (2nd - 98th percentile)
# ---------------------------
# collect x,y arrays (projected)
xs = pd.concat([
    gdf_o_web.geometry.x.rename('x') if not gdf_o_web.empty else pd.Series(dtype=float),
    gdf_d_web.geometry.x.rename('x') if not gdf_d_web.empty else pd.Series(dtype=float)
], ignore_index=True)
ys = pd.concat([
    gdf_o_web.geometry.y.rename('y') if not gdf_o_web.empty else pd.Series(dtype=float),
    gdf_d_web.geometry.y.rename('y') if not gdf_d_web.empty else pd.Series(dtype=float)
], ignore_index=True)

if len(xs) == 0 or len(ys) == 0 or not np.isfinite(xs.to_numpy()).any():
    raise RuntimeError('No valid projected coordinates found to compute map extent.')

low_pct, high_pct = 2, 98
minx, maxx = np.percentile(xs, [low_pct, high_pct])
miny, maxy = np.percentile(ys, [low_pct, high_pct])

# fallback to full min/max if needed
full_minx, full_maxx = xs.min(), xs.max()
full_miny, full_maxy = ys.min(), ys.max()

# add padding (at least 300 m)
span_x = maxx - minx
span_y = maxy - miny
pad_x = max(span_x * 0.06, 300)
pad_y = max(span_y * 0.06, 300)
extent = (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y)

# safety clamp if extent absurdly large
max_allowed_span = 200_000  # 200 km
if (extent[1] - extent[0] > max_allowed_span) or (extent[3] - extent[2] > max_allowed_span):
    print('WARNING: computed extent is very large. Falling back to full bounds.')
    pad_x_f = max((full_maxx - full_minx) * 0.06, 300)
    pad_y_f = max((full_maxy - full_miny) * 0.06, 300)
    extent = (full_minx - pad_x_f, full_maxx + pad_x_f, full_miny - pad_y_f, full_maxy + pad_y_f)

# print sample outliers (stations outside percentile window) to help debugging
outlier_mask_o = (gdf_o_web.geometry.x < minx) | (gdf_o_web.geometry.x > maxx) | (gdf_o_web.geometry.y < miny) | (gdf_o_web.geometry.y > maxy)
outlier_mask_d = (gdf_d_web.geometry.x < minx) | (gdf_d_web.geometry.x > maxx) | (gdf_d_web.geometry.y < miny) | (gdf_d_web.geometry.y > maxy)
outliers_o = gdf_o_web.loc[outlier_mask_o, ['station_id', 'lat', 'lon']].drop_duplicates().head(10)
outliers_d = gdf_d_web.loc[outlier_mask_d, ['station_id', 'lat', 'lon']].drop_duplicates().head(10)
if not outliers_o.empty or not outliers_d.empty:
    print('Found station coordinates outside the central 2-98 percentile (sample, up to 10 shown each):')
    if not outliers_o.empty:
        print(' Origin outliers:')
        print(outliers_o.to_string(index=False))
    if not outliers_d.empty:
        print(' Destination outliers:')
        print(outliers_d.to_string(index=False))

# ---------------------------
# plotting helpers
# ---------------------------
def sizes_from_counts(series, base=POINT_BASE):
    arr = np.sqrt(series.fillna(0).astype(float) + 1.0)
    if arr.max() > 0:
        scaled = base * (arr / arr.max())
    else:
        scaled = np.full_like(arr, base * 0.2)
    return np.clip(scaled, 2, base * 1.1)

# main loop: produce map PNG and a separate bar PNG for each hour
for h in range(24):
    o_h = gdf_o_web[gdf_o_web['hour'] == h].copy()
    d_h = gdf_d_web[gdf_d_web['hour'] == h].copy()

    # compute maximum count for color scaling
    max_count = int(max(o_h['count'].max() if not o_h.empty else 0,
                        d_h['count'].max() if not d_h.empty else 0, 1))

    # ---------- MAP FIGURE (orig left, dest right) ----------
    fig, (ax_o, ax_d) = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)

    # set extent and add basemap first (so tiles cover area)
    for ax in (ax_o, ax_d):
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    if use_ctx:
        try:
            import contextily as ctx
            ctx.add_basemap(ax_o, crs=gdf_o_web.crs.to_string(), source=ctx.providers.CartoDB.Positron)
            ctx.add_basemap(ax_d, crs=gdf_d_web.crs.to_string(), source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            print('contextily error (continuing without basemap):', e)

    # faint background of all stations (for context)
    if not gdf_o_web.empty:
        ax_o.scatter(gdf_o_web.geometry.x, gdf_o_web.geometry.y, s=3, color='lightgrey', alpha=0.6, zorder=0)
        ax_d.scatter(gdf_o_web.geometry.x, gdf_o_web.geometry.y, s=3, color='lightgrey', alpha=0.6, zorder=0)

    # plot origins (red)
    if not o_h.empty:
        sizes_o = sizes_from_counts(o_h['count'])
        cmap_o = plt.colormaps.get_cmap(CMAP_ORIG)
        norm_o = Normalize(vmin=0, vmax=max_count)
        colors_o = cmap_o(norm_o(o_h['count'].astype(float)))
        ax_o.scatter(o_h.geometry.x, o_h.geometry.y, s=sizes_o, color=colors_o, edgecolors='k', linewidth=0.2, zorder=3)
        sm_o = cm.ScalarMappable(norm=norm_o, cmap=CMAP_ORIG)
        sm_o._A = []
        fig.colorbar(sm_o, ax=ax_o, fraction=0.046, pad=0.02).set_label('Origin count (hour)')

    ax_o.set_title(f'Origins — hour {h}')
    ax_o.axis('off')

    # plot destinations (blue)
    if not d_h.empty:
        sizes_d = sizes_from_counts(d_h['count'])
        cmap_d = plt.colormaps.get_cmap(CMAP_DEST)
        norm_d = Normalize(vmin=0, vmax=max_count)
        colors_d = cmap_d(norm_d(d_h['count'].astype(float)))
        ax_d.scatter(d_h.geometry.x, d_h.geometry.y, s=sizes_d, color=colors_d, edgecolors='k', linewidth=0.2, zorder=3)
        sm_d = cm.ScalarMappable(norm=norm_d, cmap=CMAP_DEST)
        sm_d._A = []
        fig.colorbar(sm_d, ax=ax_d, fraction=0.046, pad=0.02).set_label('Destination count (hour)')

    ax_d.set_title(f'Destinations — hour {h}')
    ax_d.axis('off')

    map_out = MAP_DIR / f'hour_{h:02d}_maps.png'
    fig.savefig(map_out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # ---------- BARCHART FIGURE (separate) ----------
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)

    if not o_h.empty:
        top_o = o_h.sort_values('count', ascending=False).head(TOP_K_BARS)
        labels_o = top_o['station_id'].astype(str).values[::-1]   # reversed for horizontal bars
        counts_o = top_o['count'].values[::-1]
        ax1.barh(labels_o, counts_o, color='tab:red')
        ax1.set_title(f'Top {TOP_K_BARS} origin stations (hour {h})')
        ax1.set_xlabel('Starts')
    else:
        ax1.text(0.5, 0.5, 'No origin data', ha='center', va='center')
        ax1.set_axis_off()

    if not d_h.empty:
        top_d = d_h.sort_values('count', ascending=False).head(TOP_K_BARS)
        labels_d = top_d['station_id'].astype(str).values[::-1]
        counts_d = top_d['count'].values[::-1]
        ax2.barh(labels_d, counts_d, color='tab:blue')
        ax2.set_title(f'Top {TOP_K_BARS} destination stations (hour {h})')
        ax2.set_xlabel('Ends')
    else:
        ax2.text(0.5, 0.5, 'No destination data', ha='center', va='center')
        ax2.set_axis_off()

    bars_out = BARS_DIR / f'hour_{h:02d}_bars.png'
    fig2.savefig(bars_out, dpi=DPI, bbox_inches='tight')
    plt.close(fig2)

print('Saved maps to', MAP_DIR)
print('Saved bar charts to', BARS_DIR)

#%%
# contiguous_hour_clustering_and_gravity_avg_per_day.py
"""
Same clustering + gravity script you provided, but changes to produce
OD aggregated per cluster *averaged per calendar day*.

Key change: after summing hourly OD matrices for a cluster we divide
by `n_days` (the number of unique calendar dates present in the
`combined` DataFrame). This turns cluster totals (total trips across
all days in the dataset occurring during those cluster hours) into
an average number of trips per calendar day for that cluster.

Notes / caveats:
 - `n_days` is computed globally from `combined['started_at'].dt.date.nunique()`.
   This is the simplest and most defensible choice: it gives average trips
   per calendar day across the whole observation window. If you prefer to
   compute `n_days` per-hour or per-cluster (e.g. count of days that actually
   contained at least one trip in the cluster hours), see the comment below
   and I can provide that variant.
 - The averaged `agg` (OD) is then used to compute O and D (origin/destination
   totals) and run the gravity model. The gravity model will therefore predict
   average trips per calendar day for that cluster period.

Run this in the same session where `combined` (the concatenated trip DataFrame)
exists in memory.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix

# ---------- USER CONFIG ----------
OUTPUT_DIR = Path(r'E:/Uni_PGT/visualisation_outputs/clustered_gravity_avg_per_day')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_COUNT = 4            # number of contiguous clusters to form
PCA_VARIANCE = 0.90          # keep PCA components explaining this fraction of variance
BETA_1 = 0.0005
ERROR_THRESHOLD = 0.01
IMPROVEMENT_THRESHOLD = 1e-6
MAX_ITERS = 2000
EPS = 1e-12
# ---------------------------------

# Ensure combined exists
try:
    combined
except NameError:
    raise RuntimeError('DataFrame `combined` not found in memory. Load it first as `combined`.')

# Prepare datetime and hour
combined['started_at'] = pd.to_datetime(combined['started_at'], errors='coerce')
combined = combined.dropna(subset=['started_at'])
if 'hour' not in combined.columns:
    combined['hour'] = combined['started_at'].dt.hour

# Compute number of calendar days present in the dataset (global)
# This is used to convert cluster totals -> average trips per calendar day
n_days = combined['started_at'].dt.date.nunique()
print(f"Dataset spans {n_days} calendar days (unique dates) — cluster OD will be averaged per calendar day.")

# Optional alternative (commented): compute n_days_per_cluster by counting unique dates
# with at least one trip in cluster hours. This can yield slightly different averages
# that only count days where cluster hours had any activity. If you prefer that,
# uncomment and use the per-cluster approach shown later in a comment.

# Precompute hourly OD pivot tables (raw counts per hour)
hourly_pivots = {}
for h in range(24):
    sub = combined[combined['hour'] == h]
    if sub.empty:
        hourly_pivots[h] = pd.DataFrame()
        continue
    counts = sub.groupby(['start_station_id', 'end_station_id']).size().reset_index(name='count')
    pivot = counts.pivot(index='start_station_id', columns='end_station_id', values='count').fillna(0)
    hourly_pivots[h] = pivot

# Build aligned feature vectors for each hour using union of station ids
hours = sorted(hourly_pivots.keys())
all_stations = sorted({int(s) for h in hours for s in (list(hourly_pivots[h].index) + list(hourly_pivots[h].columns))})
if len(all_stations) == 0:
    raise RuntimeError('No station IDs found in hourly pivots--check combined data')

def pivot_to_aligned_vector(pivot_df, stations):
    mat = pd.DataFrame(0.0, index=stations, columns=stations)
    if not pivot_df.empty:
        tmp = pivot_df.reindex(index=stations, columns=stations).fillna(0)
        mat.iloc[:, :] = tmp.values
    return mat.values.flatten()

X_list = []
for h in hours:
    vec = pivot_to_aligned_vector(hourly_pivots[h], all_stations)
    X_list.append(vec)
X = np.vstack(X_list)  # shape (24, n_features)

# Scale + PCA
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=PCA_VARIANCE, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)

# Build chain connectivity (adjacency) for contiguous clustering
n_hours = X_pca.shape[0]
rows = []
cols = []
data = []
for i in range(n_hours - 1):
    rows.extend([i, i+1])
    cols.extend([i+1, i])
    data.extend([1, 1])
connectivity = csr_matrix((data, (rows, cols)), shape=(n_hours, n_hours))

# Run contiguous agglomerative clustering
agg = AgglomerativeClustering(n_clusters=CLUSTER_COUNT, linkage='ward', connectivity=connectivity)
labels = agg.fit_predict(X_pca)

hour_cluster_df = pd.DataFrame({'hour': hours, 'cluster': labels}).sort_values('hour')
hour_cluster_df.to_csv(OUTPUT_DIR / 'cluster_membership.csv', index=False)
print('Hour -> Cluster mapping saved to', OUTPUT_DIR / 'cluster_membership.csv')

# Helper: haversine pairwise
def haversine_pairwise(lons, lats):
    R = 6371000.0
    lon = np.radians(lons)
    lat = np.radians(lats)
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat/2.0)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c

# Impedance function (single det1 used here)
def new_cost1(cost_matrix, beta=BETA_1):
    return np.exp(-beta * cost_matrix)

# Gravity model (doubly-constrained IPF)
def gravity_model(O, D, det, error_threshold=ERROR_THRESHOLD, improvement_threshold=IMPROVEMENT_THRESHOLD, max_iters=MAX_ITERS):
    O = np.array(O, dtype=float).copy()
    D = np.array(D, dtype=float).copy()
    sum_O = O.sum(); sum_D = D.sum()
    if sum_O <= 0 or sum_D <= 0:
        raise ValueError('Origin or Destination totals sum to zero')
    if abs(sum_O - sum_D) > 1e-9:
        D = D * (sum_O / sum_D)
    n = len(O)
    Ai = np.ones(n)
    Bj = np.ones(n)
    prev_error = np.inf
    Tij = np.zeros((n, n), dtype=float)
    det_mat = np.array(det, dtype=float).copy()
    det_mat[det_mat < EPS] = EPS
    iteration = 0
    while iteration < max_iters:
        iteration += 1
        denom_i = (det_mat * (Bj * D)[None, :]).sum(axis=1) + EPS
        Ai = 1.0 / denom_i
        denom_j = (det_mat * (Ai * O)[:, None]).sum(axis=0) + EPS
        Bj_new = 1.0 / denom_j
        Tij = (Ai * O)[:, None] * (Bj_new * D)[None, :] * det_mat
        error = (np.abs(O - Tij.sum(axis=1)).sum() + np.abs(D - Tij.sum(axis=0)).sum()) / (sum_O + EPS)
        improvement = abs(prev_error - error)
        if error < error_threshold:
            stop_reason = 'Error threshold met'
            break
        if improvement < improvement_threshold:
            stop_reason = 'Slow improvement'
            break
        prev_error = error
        Bj = Bj_new
    else:
        stop_reason = 'max_iters'
    diagnostics = {'iterations': iteration, 'error': float(error), 'stop_reason': stop_reason}
    return Tij, diagnostics

# Metrics
def calculate_metrics(predicted_T, observed_T_df):
    obs = observed_T_df.to_numpy().astype(float)
    pred = np.array(predicted_T, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError('Predicted and observed shapes differ')
    obs_f = obs.flatten(); pred_f = pred.flatten()
    mse = np.mean((obs_f - pred_f) ** 2)
    rmse = float(np.sqrt(mse))
    ss_tot = np.sum((obs_f - obs_f.mean()) ** 2)
    ss_res = np.sum((obs_f - pred_f) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + EPS)))
    return {'rmse': rmse, 'r2': r2}

# Aggregate OD per cluster and run gravity — now averaging per calendar day
for c in sorted(hour_cluster_df['cluster'].unique()):
    hrs = hour_cluster_df[hour_cluster_df['cluster'] == c]['hour'].tolist()
    print(f'Processing cluster {c}: hours = {hrs}')
    # sum OD counts across hours in cluster
    agg = None
    for h in hrs:
        p = hourly_pivots[h]
        if agg is None:
            agg = p.copy()
        else:
            agg = agg.add(p, fill_value=0)
    agg = agg.fillna(0)

    # ----- NEW: convert cluster totals -> average per calendar day -----
    # Divide the aggregated matrix by the number of calendar days in the dataset
    # so that values represent "average trips per calendar day during the cluster hours".
    if n_days > 0:
        agg = agg / float(n_days)
    else:
        print('Warning: n_days==0, skipping division (no date information)')
    # ------------------------------------------------------------------

    if agg.empty:
        print(f'Cluster {c}: empty aggregated OD, skipping')
        continue

    # ensure square by intersection
    rows = [int(x) for x in agg.index.tolist()]
    cols = [int(x) for x in agg.columns.tolist()]
    common = sorted(list(set(rows) & set(cols)))
    if len(common) < 2:
        print(f'Cluster {c}: too few common stations ({len(common)}), skipping')
        continue
    agg = agg.reindex(index=common, columns=common).fillna(0)

    # Save the averaged cluster OD (this CSV now contains average trips per calendar day for the cluster hours)
    agg.to_csv(OUTPUT_DIR / f'cluster_od_avgperday_{c}.csv')

    n = len(common)
    print(f'Cluster {c}: n_stations={n} (averaged over {n_days} days)')
    if n > 1500:
        print('WARNING: cluster has many stations (>1500); this may be slow and memory-heavy')

    # build coord df (median of start/end coords)
    lon_cols = [col for col in combined.columns if 'longitude' in col.lower()]
    lat_cols = [col for col in combined.columns if 'latitude' in col.lower()]
    start_lon = next((col for col in lon_cols if col.lower().startswith('start')), None)
    start_lat = next((col for col in lat_cols if col.lower().startswith('start')), None)
    end_lon = next((col for col in lon_cols if col.lower().startswith('end')), None)
    end_lat = next((col for col in lat_cols if col.lower().startswith('end')), None)
    coord_parts = []
    if start_lon and start_lat:
        tmp = combined.groupby('start_station_id')[[start_lon, start_lat]].median()
        tmp = tmp.rename(columns={start_lon:'lon', start_lat:'lat'})
        coord_parts.append(tmp)
    if end_lon and end_lat:
        tmp2 = combined.groupby('end_station_id')[[end_lon, end_lat]].median()
        tmp2 = tmp2.rename(columns={end_lon:'lon', end_lat:'lat'})
        coord_parts.append(tmp2)
    if not coord_parts:
        raise RuntimeError('Could not find station lon/lat columns in combined')
    coord_df = pd.concat(coord_parts).groupby(level=0).first()
    coord_df = coord_df.reindex(common).dropna()
    if len(coord_df) != n:
        missing = set(common) - set(coord_df.index.astype(int).tolist())
        if missing:
            print(f'Cluster {c}: dropping {len(missing)} stations with missing coords (sample): {list(missing)[:10]}')
            keep = [s for s in common if s not in missing]
            if len(keep) < 2:
                print(f'Cluster {c}: too few stations after dropping, skipping')
                continue
            agg = agg.reindex(index=keep, columns=keep).fillna(0)
            coord_df = coord_df.reindex(keep)
            common = keep
            n = len(common)

    # compute cost matrix
    lons = coord_df['lon'].to_numpy(dtype=float)
    lats = coord_df['lat'].to_numpy(dtype=float)
    cost_m = haversine_pairwise(lons, lats)
    pd.DataFrame(cost_m, index=common, columns=common).to_csv(OUTPUT_DIR / f'cost_matrix_cluster_{c}.csv')

    # deterrence matrix
    det1 = new_cost1(cost_m, beta=BETA_1)

    # totals (these totals are now average trips per calendar day for the cluster hours)
    O = agg.sum(axis=1).to_numpy()
    D = agg.sum(axis=0).to_numpy()

    # run gravity
    Tij1, diag1 = gravity_model(O.copy(), D.copy(), det1)
    pred1_df = pd.DataFrame(Tij1, index=agg.index, columns=agg.columns)
    pred1_df.to_csv(OUTPUT_DIR / f'predicted_gravity_det1_cluster_{c}_avgperday.csv')
    metrics1 = calculate_metrics(pred1_df, agg)

    pd.DataFrame([diag1]).to_csv(OUTPUT_DIR / f'diagnostics_det1_cluster_{c}.csv', index=False)
    pd.DataFrame([metrics1]).to_csv(OUTPUT_DIR / f'metrics_det1_cluster_{c}.csv', index=False)

    print(f'Cluster {c}: done. metrics_det1={metrics1}')

print('All clusters processed. Outputs in', OUTPUT_DIR)

#%%
# beta_grid_det1_avg_per_day.py
# Updated beta-grid search script that expects cluster-aggregated OD to be
# averaged per calendar day before running gravity model.
# This version computes n_days from the `combined` DataFrame and divides the
# aggregated cluster OD by n_days (so O/D represent average trips per calendar day).

import numpy as np
import pandas as pd
from pathlib import Path

# ---------- USER / RUN-TIME CONFIG ----------
beta_grid = np.logspace(-6, -2, 25)   # grid to search for beta (adjust if desired)
OUTPUT_DIR = Path(r'E:/Uni_PGT/visualisation_outputs/clustered_gravity')  # match your cluster script
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EPS = 1e-12
# --------------------------------------------

# Basic haversine (meters)
def haversine_pairwise(lons, lats):
    R = 6371000.0
    lon = np.radians(lons)
    lat = np.radians(lats)
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat/2.0)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c

# Normalized RMSE helper
def normalized_rmse(pred_df, obs_df):
    obs = obs_df.to_numpy(dtype=float)
    pred = pred_df.to_numpy(dtype=float)
    mse = np.mean((obs - pred) ** 2)
    rmse = float(np.sqrt(mse))
    mean_obs = float(np.mean(obs))
    if mean_obs == 0:
        return {'rmse': rmse, 'nrmse': float('inf')}
    return {'rmse': rmse, 'nrmse': float(rmse / mean_obs)}

# Ensure gravity_model & calculate_metrics exist (from your previous script). If not, provide safe fallback.
try:
    gravity_model  # noqa
except NameError:
    raise RuntimeError("gravity_model is not defined in the session. Run the clustering/gravity script first (which defines gravity_model).")

try:
    calculate_metrics  # noqa
except NameError:
    # fallback basic metrics (should be similar to your earlier calculate_metrics)
    def calculate_metrics(pred_df, obs_df):
        obs = obs_df.to_numpy(dtype=float)
        pred = pred_df.to_numpy(dtype=float)
        if obs.shape != pred.shape:
            raise ValueError('Predicted and observed shapes differ')
        obs_f = obs.flatten()
        pred_f = pred.flatten()
        mse = np.mean((obs_f - pred_f) ** 2)
        rmse = float(np.sqrt(mse))
        ss_total = np.sum((obs_f - obs_f.mean()) ** 2)
        ss_residual = np.sum((obs_f - pred_f) ** 2)
        r_squared = float(1.0 - (ss_residual / (ss_total + EPS)))
        return {'rmse': rmse, 'r2': r_squared}

# Core function: runs grid search for det1 on a single aggregated cluster
def run_beta_grid_on_cluster(agg_df, coord_df, cost_func, det_func_factory, beta_values, gravity_model_func, metrics_func):
    """
    Returns best = {'beta', 'metrics', 'pred_df', 'diag'} using primary=r2, secondary=-nrmse
    """
    common = [int(x) for x in agg_df.index.tolist()]
    lons = coord_df.loc[common, 'lon'].to_numpy(dtype=float)
    lats = coord_df.loc[common, 'lat'].to_numpy(dtype=float)
    cost_m = cost_func(lons, lats)

    O = agg_df.sum(axis=1).to_numpy()
    D = agg_df.sum(axis=0).to_numpy()

    best = {'beta': None, 'metrics': None, 'pred_df': None, 'diag': None}
    best_score = None

    for beta in beta_values:
        det = det_func_factory(cost_m, beta=beta)
        try:
            Tij, diag = gravity_model_func(O.copy(), D.copy(), det)
        except Exception:
            # solver failure for this beta — skip
            continue
        pred_df = pd.DataFrame(Tij, index=agg_df.index, columns=agg_df.columns)
        mets = metrics_func(pred_df, agg_df)    # should include 'r2'
        nr = normalized_rmse(pred_df, agg_df)
        mets.update(nr)
        score = (mets.get('r2', -9999), - (mets.get('nrmse', np.inf) if np.isfinite(mets.get('nrmse', np.inf)) else np.inf))
        if best_score is None or score > best_score:
            best_score = score
            best = {'beta': float(beta), 'metrics': mets, 'pred_df': pred_df.copy(), 'diag': diag}
    return best

# DET1 factory (exponential), no det2 per your request
def det1_factory(costm, beta):
    return np.exp(-beta * costm)

# Containers for results
best_betas = {}          # {cluster: {'beta':..., 'metrics':..., 'diag':...}}
predicted_matrices = {}  # {cluster: predicted_df}

# Compute number of calendar days in the combined dataset (used to convert cluster totals -> avg per calendar day)
try:
    n_days = int(combined['started_at'].dt.date.nunique())
except Exception:
    # fallback: if combined not present or started_at not parsed, set to 1 (no scaling)
    n_days = 1

clusters = sorted(hour_cluster_df['cluster'].unique())

for c in clusters:
    hrs = hour_cluster_df[hour_cluster_df['cluster']==c]['hour'].tolist()
    # aggregate OD across hours in this cluster
    agg = None
    for h in hrs:
        p = hourly_pivots[h]
        if agg is None:
            agg = p.copy()
        else:
            agg = agg.add(p, fill_value=0)
    if agg is None:
        print(f'Cluster {c}: empty aggregation, skipping')
        continue
    agg = agg.fillna(0)

    # --- NEW: convert aggregated cluster totals to average trips per calendar day ---
    if n_days > 1:
        agg = agg / float(n_days)
    # save the averaged cluster OD for diagnostic purposes
    agg.to_csv(OUTPUT_DIR / f'cluster_od_avg_per_day_{c}.csv')

    # square matrix by intersection of rows & cols
    rows = [int(x) for x in agg.index.tolist()]
    cols = [int(x) for x in agg.columns.tolist()]
    common = sorted(list(set(rows) & set(cols)))
    if len(common) < 2:
        print(f'Cluster {c} skipped (too few common stations)')
        continue
    agg = agg.reindex(index=common, columns=common).fillna(0)

    # build coordinate DF for these common stations (median of start/end coords)
    lon_cols = [col for col in combined.columns if 'longitude' in col.lower()]
    lat_cols = [col for col in combined.columns if 'latitude' in col.lower()]
    start_lon = next((col for col in lon_cols if col.lower().startswith('start')), None)
    start_lat = next((col for col in lat_cols if col.lower().startswith('start')), None)
    end_lon = next((col for col in lon_cols if col.lower().startswith('end')), None)
    end_lat = next((col for col in lat_cols if col.lower().startswith('end')), None)
    coord_parts = []
    if start_lon and start_lat:
        tmp = combined.groupby('start_station_id')[[start_lon, start_lat]].median()
        tmp = tmp.rename(columns={start_lon:'lon', start_lat:'lat'})
        coord_parts.append(tmp)
    if end_lon and end_lat:
        tmp2 = combined.groupby('end_station_id')[[end_lon, end_lat]].median()
        tmp2 = tmp2.rename(columns={end_lon:'lon', end_lat:'lat'})
        coord_parts.append(tmp2)
    if not coord_parts:
        raise RuntimeError('Could not find station longitude/latitude columns in `combined`.')

    coord_df = pd.concat(coord_parts).groupby(level=0).first()
    coord_df = coord_df.reindex(common).dropna()

    if len(coord_df) != len(common):
        missing = set(common) - set(coord_df.index.astype(int).tolist())
        print(f'Cluster {c}: dropping {len(missing)} stations missing coords (sample): {list(missing)[:6]}')
        keep = [s for s in common if s not in missing]
        agg = agg.reindex(index=keep, columns=keep).fillna(0)
        coord_df = coord_df.reindex(keep)
        common = keep
        if len(common) < 2:
            print(f'Cluster {c}: too few stations after dropping coords, skipping')
            continue

    # run grid search for det1 only
    best1 = run_beta_grid_on_cluster(agg, coord_df, haversine_pairwise, det1_factory, beta_grid, gravity_model, calculate_metrics)

    # save best1 outputs
    if best1['beta'] is None:
        print(f'Cluster {c}: no successful beta found (solver failed for all candidates)')
        continue

    best_betas[c] = {'beta_det1': best1['beta'], 'metrics_det1': best1['metrics'], 'diag_det1': best1['diag']}
    predicted_matrices[c] = best1['pred_df']

    # persist to disk
    best1['pred_df'].to_csv(OUTPUT_DIR / f'best_pred_det1_clusterr_{c}.csv')
    pd.DataFrame([best1['metrics']]).to_csv(OUTPUT_DIR / f'best_metrics_det1_clusterr_{c}.csv', index=False)

    print(f"Cluster {c}: done. best_beta_det1={best1['beta']:.2e}, metrics={best1['metrics']}")

# save summary CSV
summary = []
for c, info in best_betas.items():
    summary.append({'cluster': c, 'best_beta_det1': info['beta_det1'], **{f"metrics_{k}": v for k,v in info['metrics_det1'].items()}})
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUTPUT_DIR / 'beta_gridsearch_summary_det1_only.csv', index=False)

print('Grid search (det1 only) completed. Summary saved to:', OUTPUT_DIR / 'beta_gridsearch_summary_det1_only.csv')
print('Best betas dict: variable `best_betas` (in memory).')
print('Predicted OD DataFrames per cluster: variable `predicted_matrices` (in memory).')
#%%
"""
station_to_poi_od_from_predicted_fixed.py

Fixed version of the POI-OD conversion that uses gravity-model predicted
station->station OD matrices (per cluster), and avoids the ValueError you ran
into by correctly sizing the POI index for each output matrix.

Assumptions:
 - `predicted_matrices` is a dict in memory mapping cluster -> pandas.DataFrame
   with station IDs as index and columns (square) containing predicted Tij counts.
 - `combined` DataFrame is in memory (for station coords).
 - POIs CSV is at POI_FILE and contains columns ['lat','lon','category'] and
   optionally 'poi_id'.

Saves per-cluster POI OD CSVs (only for POIs that receive >0 allocation for
stations in that cluster), plus an aggregated POI OD across clusters.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---------- USER CONFIG ----------
POI_FILE = r'E:\Uni_PGT\Express\edinburgh_pois.csv'
OUTPUT_DIR = Path(r'E:\Uni_PGT\visualisation_outputs\poi_od_fixed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_NEIGHBORS = 3
CLOSE_RADIUS_METERS = 500
CATEGORY_WEIGHTS = {
    'library': 1.1,
    'school': 1.3,
    'university': 1.5,
    'residential': 1.0,
    'commercial': 1.2,
    'hospital': 1.4
}
EPS = 1e-12
# ---------------------------------

# --- sanity checks: required in-memory objects ---
try:
    predicted_matrices
except NameError:
    raise RuntimeError("`predicted_matrices` not found in memory. Run gravity grid search first.")

try:
    combined
except NameError:
    raise RuntimeError("`combined` not found in memory. Load your combined trips dataframe.")

# --- load POIs ---
pois = pd.read_csv(POI_FILE)
pois = pois.dropna(subset=['lat', 'lon']).reset_index(drop=True)
if 'poi_id' not in pois.columns:
    pois['poi_id'] = pois.index.astype(int)
pois['category'] = pois['category'].astype(str).fillna('commercial').str.lower().str.strip()
pois['weight'] = pois['category'].map(CATEGORY_WEIGHTS).fillna(1.0)

# --- build station coords (median of start/end) ---
lon_cols = [c for c in combined.columns if 'longitude' in c.lower()]
lat_cols = [c for c in combined.columns if 'latitude' in c.lower()]
start_lon = next((c for c in lon_cols if c.lower().startswith('start')), None)
start_lat = next((c for c in lat_cols if c.lower().startswith('start')), None)
end_lon = next((c for c in lon_cols if c.lower().startswith('end')), None)
end_lat = next((c for c in lat_cols if c.lower().startswith('end')), None)

coord_parts = []
if start_lon and start_lat:
    tmp = combined.groupby('start_station_id')[[start_lon, start_lat]].median()
    tmp = tmp.rename(columns={start_lon: 'lon', start_lat: 'lat'})
    coord_parts.append(tmp)
if end_lon and end_lat:
    tmp2 = combined.groupby('end_station_id')[[end_lon, end_lat]].median()
    tmp2 = tmp2.rename(columns={end_lon: 'lon', end_lat: 'lat'})
    coord_parts.append(tmp2)
if not coord_parts:
    raise RuntimeError('No station lon/lat columns found in `combined`.')

stations_df = pd.concat(coord_parts).groupby(level=0).first()
stations_df = stations_df.rename_axis('station_id')
stations_df.index = stations_df.index.astype(int)
# restrict to stations present in any predicted matrix to avoid waste
pred_station_ids = set()
for df in predicted_matrices.values():
    pred_station_ids.update([int(x) for x in df.index.astype(int).tolist()])
stations_df = stations_df.reindex(sorted(pred_station_ids)).dropna()
if stations_df.empty:
    raise RuntimeError('No station coordinates available for stations present in predicted_matrices.')
station_ids = stations_df.index.astype(int).tolist()

# --- helper: haversine (vectorized) ---
def haversine_matrix(lonA, latA, lonB, latB):
    R = 6371000.0
    lonA = np.radians(np.asarray(lonA, dtype=float))
    latA = np.radians(np.asarray(latA, dtype=float))
    lonB = np.radians(np.asarray(lonB, dtype=float))
    latB = np.radians(np.asarray(latB, dtype=float))
    dlon = lonA[:, None] - lonB[None, :]
    dlat = latA[:, None] - latB[None, :]
    a = np.sin(dlat/2.0)**2 + np.cos(latA)[:, None] * np.cos(latB)[None, :] * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c

# --- precompute nearest POIs for stations using haversine via sklearn's haversine metric ---
poi_coords = pois[['lat', 'lon']].to_numpy()
station_coords = stations_df[['lat', 'lon']].to_numpy()

# use scikit-learn NearestNeighbors with haversine (expects radians)
nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='haversine')
nbrs.fit(np.radians(poi_coords))
dists_r, idxs = nbrs.kneighbors(np.radians(station_coords))
# convert radian distances to meters
dists_m = dists_r * 6371000.0

# --- build allocations: station_id -> dict(poi_index -> weight) ---
allocations = {}
for i, sid in enumerate(station_ids):
    dists = dists_m[i]
    idx = idxs[i]
    # choose close ones if any inside CLOSE_RADIUS_METERS
    mask_close = dists <= CLOSE_RADIUS_METERS
    if mask_close.any():
        chosen_idx = idx[mask_close]
        chosen_dists = dists[mask_close]
    else:
        chosen_idx = idx
        chosen_dists = dists

    cat_weights = pois.loc[chosen_idx, 'weight'].to_numpy(dtype=float)
    inv_dist = 1.0 / (chosen_dists + EPS)
    raw_scores = cat_weights * inv_dist
    if raw_scores.sum() <= 0:
        weights = np.ones_like(raw_scores) / len(raw_scores)
    else:
        weights = raw_scores / raw_scores.sum()

    allocations[int(sid)] = {int(p_idx): float(w) for p_idx, w in zip(chosen_idx.tolist(), weights.tolist())}

# Save allocations (flat)
alloc_rows = []
for s, pdict in allocations.items():
    for pidx, w in pdict.items():
        alloc_rows.append({'station_id': int(s), 'poi_index': int(pidx), 'poi_id': int(pois.at[pidx, 'poi_id']), 'weight': float(w)})
alloc_df = pd.DataFrame(alloc_rows)
alloc_df.to_csv(OUTPUT_DIR / 'station_poi_allocations.csv', index=False)
print('Saved station->POI allocation table to', OUTPUT_DIR / 'station_poi_allocations.csv')

# --- Convert predicted station OD -> POI OD per cluster (using only used POIs per cluster) ---
poi_od_per_cluster = {}
for c, pred_df in predicted_matrices.items():
    # ensure pred_df index/cols are ints
    pred_df = pred_df.copy()
    pred_df.index = pred_df.index.astype(int)
    pred_df.columns = pred_df.columns.astype(int)

    # intersect stations with allocations
    common_stations = sorted(list(set(pred_df.index.tolist()) & set(allocations.keys())))
    if len(common_stations) < 2:
        print(f'Cluster {c}: too few stations with allocations ({len(common_stations)}), skipping')
        continue

    # build station order and Tij matrix accordingly
    Tij = pred_df.reindex(index=common_stations, columns=common_stations).fillna(0).to_numpy(dtype=float)

    # build station->poi allocation matrix A (nS x nP_used)
    # find all POI indices used by these stations
    poi_indices_used = sorted({pidx for s in common_stations for pidx in allocations[s].keys()})
    if len(poi_indices_used) == 0:
        print(f'Cluster {c}: no POIs assigned to these stations, skipping')
        continue

    nS = len(common_stations)
    nP = len(poi_indices_used)
    A = np.zeros((nS, nP), dtype=float)
    station_to_row = {s:i for i,s in enumerate(common_stations)}
    poi_to_col = {p:i for i,p in enumerate(poi_indices_used)}

    for s in common_stations:
        r = station_to_row[s]
        for pidx, w in allocations[s].items():
            if pidx in poi_to_col:
                A[r, poi_to_col[pidx]] = w

    # compute POI OD: P = A^T * Tij * A
    P = A.T.dot(Tij).dot(A)

    # build DataFrame: rows/cols labelled by poi_id (not global pois index)
    poi_ids = [int(pois.at[pidx, 'poi_id']) for pidx in poi_indices_used]
    poi_od_df = pd.DataFrame(P, index=poi_ids, columns=poi_ids)
    poi_od_per_cluster[c] = poi_od_df
    poi_od_df.to_csv(OUTPUT_DIR / f'poi_od_cluster_{c}F.csv')
    print(f'Cluster {c}: saved POI OD with shape {poi_od_df.shape} (n_pois={len(poi_ids)})')

# --- aggregate across clusters (align on poi_id union) ---
if poi_od_per_cluster:
    all_poi_ids = sorted({pid for df in poi_od_per_cluster.values() for pid in df.index.tolist()})
    agg_mat = pd.DataFrame(0.0, index=all_poi_ids, columns=all_poi_ids)
    for c, df in poi_od_per_cluster.items():
        agg_mat = agg_mat.add(df.reindex(index=all_poi_ids, columns=all_poi_ids, fill_value=0), fill_value=0)
    agg_mat.to_csv(OUTPUT_DIR / 'poi_od_aggregated_all_clusters_F.csv')
    print('Saved aggregated POI OD for all clusters to', OUTPUT_DIR / 'poi_od_aggregated_all_clusters.csv')
else:
    print('No POI OD outputs (no clusters had usable data).')

print('Done. Outputs in', OUTPUT_DIR)

#%%
# poi_od_sparse_with_baseline.py
#
# Memory-efficient conversion of predicted station->station OD (per cluster)
# into POI->POI OD using sparse matrices and a "baseline-to-k-nearest" strategy
# for neglected POIs so we avoid creating a dense 33k x 33k matrix.
#
# Requirements: scipy, scikit-learn, pandas, numpy
#
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

# ---------- USER CONFIG ----------
POI_FILE = r'E:\Uni_PGT\Express\edinburgh_pois.csv'
OUTPUT_DIR = Path(r'E:\Uni_PGT\visualisation_outputs\poi_od_sparse_baseline')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_NEIGHBORS = 3                # used for station->poi allocation
K_BASELINE_NEIGH = 10          # number of POIs to distribute baseline for neglected POI
CLOSE_RADIUS_METERS = 500
CATEGORY_WEIGHTS = {
    'library': 1.1,
    'school': 1.3,
    'university': 1.5,
    'residential': 1.0,
    'commercial': 1.2,
    'hospital': 1.4
}
BASELINE_RATIO = 0.1   # baseline magnitude relative to mean observed cell
EPS = 1e-12
# ---------------------------------

# sanity: required objects
try:
    predicted_matrices
except NameError:
    raise RuntimeError("`predicted_matrices` not in memory. Run gravity step first.")
try:
    combined
except NameError:
    raise RuntimeError("`combined` not in memory. Load combined dataframe first.")

# load POIs
pois = pd.read_csv(POI_FILE)
pois = pois.dropna(subset=['lat', 'lon']).reset_index(drop=True)
if 'poi_id' not in pois.columns:
    pois['poi_id'] = pois.index.astype(int)
pois['category'] = pois.get('category', '').astype(str).fillna('commercial').str.lower().str.strip()
pois['weight'] = pois['category'].map(CATEGORY_WEIGHTS).fillna(1.0)

# canonical lists and lookups
all_poi_ids = pois['poi_id'].astype(int).tolist()
poi_index_to_id = {i: int(pois.at[i, 'poi_id']) for i in pois.index}
poi_id_to_index = {int(pois.at[i, 'poi_id']): i for i in pois.index}

n_all = len(all_poi_ids)
print(f"POIs loaded: {n_all} (will keep results sparse)")

# build station coords as median of start/end
lon_cols = [c for c in combined.columns if 'longitude' in c.lower()]
lat_cols = [c for c in combined.columns if 'latitude' in c.lower()]
start_lon = next((c for c in lon_cols if c.lower().startswith('start')), None)
start_lat = next((c for c in lat_cols if c.lower().startswith('start')), None)
end_lon = next((c for c in lon_cols if c.lower().startswith('end')), None)
end_lat = next((c for c in lat_cols if c.lower().startswith('end')), None)
coord_parts = []
if start_lon and start_lat:
    tmp = combined.groupby('start_station_id')[[start_lon, start_lat]].median()
    tmp = tmp.rename(columns={start_lon: 'lon', start_lat: 'lat'})
    coord_parts.append(tmp)
if end_lon and end_lat:
    tmp2 = combined.groupby('end_station_id')[[end_lon, end_lat]].median()
    tmp2 = tmp2.rename(columns={end_lon: 'lon', end_lat: 'lat'})
    coord_parts.append(tmp2)
if not coord_parts:
    raise RuntimeError('No station lon/lat columns found in combined')
stations_df = pd.concat(coord_parts).groupby(level=0).first()
stations_df.index = stations_df.index.astype(int)

# restrict to stations that appear in at least one predicted matrix
pred_station_ids = set()
for m in predicted_matrices.values():
    pred_station_ids.update([int(x) for x in m.index.astype(int).tolist()])
stations_df = stations_df.reindex(sorted(pred_station_ids)).dropna()
station_ids = stations_df.index.astype(int).tolist()
print(f"Using {len(station_ids)} stations for allocation")

# precompute station->nearest POIs (small K)
poi_coords = pois[['lat','lon']].to_numpy()
station_coords = stations_df[['lat','lon']].to_numpy()

nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='haversine')
nbrs.fit(np.radians(poi_coords))
d_radians, idxs = nbrs.kneighbors(np.radians(station_coords))
dists_m = d_radians * 6371000.0

# build allocations mapping station_id -> list of (poi_index, weight)
allocations = {}
for i, sid in enumerate(station_ids):
    dists = dists_m[i]
    idx = idxs[i]
    mask_close = dists <= CLOSE_RADIUS_METERS
    if mask_close.any():
        chosen_idx = idx[mask_close]
        chosen_dists = dists[mask_close]
    else:
        chosen_idx = idx
        chosen_dists = dists
    cat_weights = pois.loc[chosen_idx, 'weight'].to_numpy(dtype=float)
    inv_dist = 1.0 / (chosen_dists + EPS)
    raw = cat_weights * inv_dist
    if raw.sum() <= 0:
        w = np.ones_like(raw) / len(raw)
    else:
        w = raw / raw.sum()
    allocations[int(sid)] = list(zip([int(p) for p in chosen_idx.tolist()], [float(x) for x in w.tolist()]))

# Save allocations summary
alloc_rows = []
for s, lst in allocations.items():
    for pidx, w in lst:
        alloc_rows.append({'station_id': s, 'poi_index': int(pidx), 'poi_id': int(pois.at[pidx,'poi_id']), 'weight': w})
alloc_df = pd.DataFrame(alloc_rows)
alloc_df.to_csv(OUTPUT_DIR / 'station_poi_allocations.csv', index=False)
print('Saved station->POI allocations (small)')

# Build POI neighbor index for baseline distribution (k baseline neighbors on POI network)
poi_nbrs = NearestNeighbors(n_neighbors=K_BASELINE_NEIGH, metric='haversine')
poi_nbrs.fit(np.radians(poi_coords))
poi_d_rad, poi_idxs = poi_nbrs.kneighbors(np.radians(poi_coords))
poi_d_m = poi_d_rad * 6371000.0

# helper: function to add entries to sparse accumulator lists

def append_entries(rows_list, cols_list, data_list, from_poi_ids, to_poi_ids, values_matrix):
    # from_poi_ids and to_poi_ids are lists of global poi_id labels (not dataframe indices)
    for i, pid in enumerate(from_poi_ids):
        for j, qid in enumerate(to_poi_ids):
            val = values_matrix[i, j]
            if val != 0 and not np.isnan(val):
                rows_list.append(poi_id_to_index[pid])
                cols_list.append(poi_id_to_index[qid])
                data_list.append(float(val))

# main conversion loop: produce sparse COO components per cluster
from scipy.sparse import coo_matrix, save_npz
poi_od_sparse_per_cluster = {}
for c, pred_df in predicted_matrices.items():
    print(f'Processing cluster {c}...')
    pred_df = pred_df.copy()
    pred_df.index = pred_df.index.astype(int)
    pred_df.columns = pred_df.columns.astype(int)

    # stations in this predicted matrix that have allocations
    common_stations = sorted(list(set(pred_df.index.tolist()) & set(allocations.keys())))
    if len(common_stations) < 2:
        print(f' Cluster {c}: too few stations with allocations, skipping')
        continue

    Tij = pred_df.reindex(index=common_stations, columns=common_stations).fillna(0).to_numpy(dtype=float)

    # build A_used (nS x nP_used) where nP_used is number of distinct POIs assigned to these stations
    poi_indices_used = sorted({pidx for s in common_stations for pidx, _ in allocations[s]})
    if len(poi_indices_used) == 0:
        print(f' Cluster {c}: no POIs used by these stations, skipping')
        continue

    station_to_row = {s:i for i,s in enumerate(common_stations)}
    poi_to_col_used = {p:i for i,p in enumerate(poi_indices_used)}
    nS = len(common_stations); nP_used = len(poi_indices_used)
    A_used = np.zeros((nS, nP_used), dtype=float)
    for s in common_stations:
        r = station_to_row[s]
        for pidx, w in allocations[s]:
            if pidx in poi_to_col_used:
                A_used[r, poi_to_col_used[pidx]] = w

    # compute P_used (nP_used x nP_used) — this is small (few hundreds)
    P_used = A_used.T.dot(Tij).dot(A_used)

    # Now map P_used into global sparse lists
    rows = []
    cols = []
    data = []

    # map used POI local -> global poi_id
    poi_ids_used = [int(pois.at[pidx, 'poi_id']) for pidx in poi_indices_used]

    # append P_used block into sparse lists
    for i_local, global_pidx in enumerate(poi_indices_used):
        pid = int(pois.at[global_pidx, 'poi_id'])
        for j_local, global_qidx in enumerate(poi_indices_used):
            qid = int(pois.at[global_qidx, 'poi_id'])
            val = P_used[i_local, j_local]
            if val != 0 and not np.isnan(val):
                rows.append(poi_id_to_index[pid])
                cols.append(poi_id_to_index[qid])
                data.append(float(val))

    # compute baseline per-cell scalar (mean observed cell over used block)
    observed_sum = P_used.sum()
    if observed_sum <= 0:
        mean_cell = 0.0
    else:
        mean_cell = observed_sum / (n_all)  # normalized to full universe

    mean_weight = pois['weight'].mean()

    # identify neglected POIs (those that have no entries in current sparse lists)
    used_poi_set = set(poi_ids_used)
    all_poi_set = set(all_poi_ids)
    neglected = sorted(list(all_poi_set - used_poi_set))
    print(f' Cluster {c}: used POIs={len(used_poi_set)}, neglected POIs={len(neglected)}')

    # For each neglected POI, distribute baseline to its K_BASELINE_NEIGH nearest POIs (by index in pois)
    if len(neglected) > 0 and mean_cell > 0:
        # we have precomputed poi_idxs and poi_d_m arrays (POI->neighbors)
        for pid in neglected:
            pidx = poi_id_to_index[pid]
            # neighbors indices (including itself) from poi_idxs array
            neigh_local = poi_idxs[pidx, 1:K_BASELINE_NEIGH+1]  # skip self (first neighbor)
            neigh_d = poi_d_m[pidx, 1:K_BASELINE_NEIGH+1]
            # weights: inverse distance * category weight
            neigh_weights = pois.loc[neigh_local, 'weight'].to_numpy(dtype=float)
            invd = 1.0 / (neigh_d + EPS)
            raw = neigh_weights * invd
            if raw.sum() <= 0:
                wnorm = np.ones_like(raw) / len(raw)
            else:
                wnorm = raw / raw.sum()
            # baseline total per neglected POI (outflow) distribute across neighbors
            baseline_total = mean_cell * BASELINE_RATIO * (pois.at[pidx, 'weight'] / mean_weight)
            # distribute baseline_total to outgoing links pid -> neigh
            for k_idx, neigh_idx in enumerate(neigh_local):
                qid = int(pois.at[int(neigh_idx), 'poi_id'])
                val = baseline_total * wnorm[k_idx]
                rows.append(pidx)
                cols.append(neigh_idx)
                data.append(float(val))
            # similarly distribute baseline inflow: neighbors -> pid
            for k_idx, neigh_idx in enumerate(neigh_local):
                qid = int(pois.at[int(neigh_idx), 'poi_id'])
                val = baseline_total * wnorm[k_idx]
                rows.append(neigh_idx)
                cols.append(pidx)
                data.append(float(val))

    # build sparse COO and save
    coo = coo_matrix((data, (rows, cols)), shape=(n_all, n_all))
    # compress to csr for smaller memory on disk
    csr = coo.tocsr()
    save_npz(OUTPUT_DIR / f'poi_od_cluster_{c}.npz', csr)

    # also save edge-list CSV (only nonzero entries)
    coo_nz = coo.tocoo()
    edge_df = pd.DataFrame({'from_poi_index': coo_nz.row, 'to_poi_index': coo_nz.col, 'flow': coo_nz.data})
    # map poi_index -> poi_id for readability
    edge_df['from_poi_id'] = edge_df['from_poi_index'].map(lambda x: int(pois.at[int(x),'poi_id']))
    edge_df['to_poi_id'] = edge_df['to_poi_index'].map(lambda x: int(pois.at[int(x),'poi_id']))
    edge_df = edge_df[['from_poi_id','to_poi_id','flow']]
    edge_df.to_csv(OUTPUT_DIR / f'poi_od_cluster_{c}_edgelists.csv', index=False)

    poi_od_sparse_per_cluster[c] = {'sparse': csr, 'edge_csv': OUTPUT_DIR / f'poi_od_cluster_{c}_edgelist.csv'}
    print(f' Cluster {c}: saved sparse POI OD (nnz={csr.nnz})')

# aggregate across clusters (sum sparse matrices)
if poi_od_sparse_per_cluster:
    first = True
    agg_csr = None
    for c, info in poi_od_sparse_per_cluster.items():
        mat = info['sparse']
        if first:
            agg_csr = mat.copy()
            first = False
        else:
            agg_csr = agg_csr + mat
    # save aggregated
    save_npz(OUTPUT_DIR / 'poi_od_aggregated_all_clusters_sparse.npz', agg_csr)
    # also save aggregated edge list (may still be big) — write only top flows if desired
    coo_agg = agg_csr.tocoo()
    edge_agg = pd.DataFrame({'from_idx': coo_agg.row, 'to_idx': coo_agg.col, 'flow': coo_agg.data})
    edge_agg['from_poi_id'] = edge_agg['from_idx'].map(lambda x: int(pois.at[int(x),'poi_id']))
    edge_agg['to_poi_id'] = edge_agg['to_idx'].map(lambda x: int(pois.at[int(x),'poi_id']))
    
    edge_agg[['from_poi_id','to_poi_id','flow']].to_csv(OUTPUT_DIR / 'poi_od_aggregated_all_clusters_edgelistsF.csv', index=False)
    print('Saved aggregated sparse POI OD and edge list')

print('Done. Outputs (sparse .npz + edgelists) in', OUTPUT_DIR)

#%%
print(baseline_total)
#%%
# kmeans_candidates_no_snap.py
# Demand-weighted KMeans (NO snapping to POIs) to produce P candidate stations
# Inputs:
#  - POI_FILE: CSV with columns ['poi_id'(optional),'lat','lon',...]
#  - POI_EDGELIST: combined POI edgelist CSV with columns [from_poi_id,to_poi_id,flow]
# Outputs saved to OUTPUT_DIR:
#  - candidate_stations_P{P}_kmeans_no_snap.csv (centroid lon/lat + demand stats)
#  - candidate_eval_P{P}_kmeans_no_snap.csv

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# --------- USER CONFIG ----------
POI_FILE = r'E:\Uni_PGT\Express\edinburgh_pois.csv'

POI_EDGELIST = r'E:\Uni_PGT\visualisation_outputs\poi_od_sparse_baseline\poi_od_aggregated_all_clusters_edgelistsF.csv'
OUTPUT_DIR = Path(r'E:\Uni_PGT\visualisation_outputs\candidate_stations')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

P = 300                          # number of candidate centroids to produce
METHOD_NAME = 'kmeans_no_snap'
KMEANS_RANDOM_STATE = 42
KM_INIT = 'k-means++'           # 'k-means++' or 'random'
COVERAGE_RADIUS_METERS = 800.0  # evaluation radius for coverage (walk radius)
# --------------------------------

# Utility: small haversine helpers (meters)
def haversine_matrix(lonsA, latsA, lonsB, latsB):
    R = 6371000.0
    lonA = np.radians(np.asarray(lonsA, dtype=float))
    latA = np.radians(np.asarray(latsA, dtype=float))
    lonB = np.radians(np.asarray(lonsB, dtype=float))
    latB = np.radians(np.asarray(latsB, dtype=float))
    dlon = lonA[:, None] - lonB[None, :]
    dlat = latA[:, None] - latB[None, :]
    a = np.sin(dlat/2.0)**2 + np.cos(latA)[:, None] * np.cos(latB)[None, :] * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c

# --- load POIs and compute demand per POI from edgelist ---
print('Loading POIs...')
pois = pd.read_csv(POI_FILE)
pois = pois.dropna(subset=['lat', 'lon']).reset_index(drop=True)
if 'poi_id' not in pois.columns:
    pois['poi_id'] = pois.index.astype(int)
pois['poi_id'] = pois['poi_id'].astype(int)

print('Loading POI edgelist to compute demand per POI ...')
edges = pd.read_csv(POI_EDGELIST)
# detect flow column
cols_lower = {c.lower(): c for c in edges.columns}
flow_col = None
for candidate in ('flow', 'demand', 'count', 'weight'):
    if candidate in cols_lower:
        flow_col = cols_lower[candidate]
        break
if flow_col is None:
    numeric_cols = [c for c in edges.columns if np.issubdtype(edges[c].dtype, np.number)]
    if numeric_cols:
        flow_col = numeric_cols[-1]
    else:
        raise RuntimeError('Could not detect numeric flow column in edgelist.')
print('Flow column detected:', flow_col)

inflow = edges.groupby('to_poi_id')[flow_col].sum()
outflow = edges.groupby('from_poi_id')[flow_col].sum()
all_ids = sorted(set(inflow.index.tolist()) | set(outflow.index.tolist()) | set(pois['poi_id'].tolist()))

# demand = inflow + outflow (symmetric importance for being origin/destination)
demand_series = pd.Series(0.0, index=all_ids, dtype=float)
if not inflow.empty:
    demand_series.loc[inflow.index] += inflow
if not outflow.empty:
    demand_series.loc[outflow.index] += outflow

poi_demand = pois.set_index('poi_id').reindex(all_ids).copy()
poi_demand['demand'] = demand_series.reindex(poi_demand.index).fillna(0.0)

poi_ids = poi_demand.index.astype(int).tolist()
lats = poi_demand['lat'].to_numpy(dtype=float)
lons = poi_demand['lon'].to_numpy(dtype=float)
weights = poi_demand['demand'].to_numpy(dtype=float)

total_demand = weights.sum()
print(f'POIs loaded: {len(poi_ids)} POIs, total demand = {total_demand:.2f}')

# avoid zero-weight issues
weights_safe = weights.copy()
weights_safe[weights_safe <= 0] = 1e-6

# Project lon/lat to a simple local metric space (equirectangular approx) for KMeans
mean_lat = np.mean(lats)
xm = (lons - np.mean(lons)) * (111320 * np.cos(np.radians(mean_lat)))
ym = (lats - np.mean(lats)) * 110540
X_m = np.column_stack([xm, ym])

# Run weighted KMeans (sample_weight)
print('Running demand-weighted KMeans (no snapping) ...')
km = KMeans(n_clusters=P, init=KM_INIT, random_state=KMEANS_RANDOM_STATE, n_init=10)
km.fit(X_m, sample_weight=weights_safe)
centers_m = km.cluster_centers_

# Map centers back to lon/lat approx
centers_lon = (centers_m[:,0] / (111320 * np.cos(np.radians(mean_lat)))) + np.mean(lons)
centers_lat = (centers_m[:,1] / 110540) + np.mean(lats)

candidates = pd.DataFrame({'centroid_lon': centers_lon, 'centroid_lat': centers_lat})
candidates['method'] = METHOD_NAME
candidates['cluster_label_kmeans'] = np.arange(len(candidates))

# Evaluate coverage: distance from each POI to nearest centroid
D = haversine_matrix(lons, lats, candidates['centroid_lon'].to_numpy(), candidates['centroid_lat'].to_numpy())
min_dist = D.min(axis=1)
within = (min_dist <= COVERAGE_RADIUS_METERS)
covered_demand = weights[within].sum()
coverage_fraction = covered_demand / (weights.sum() + 1e-12)
avg_weighted_dist = np.sum(min_dist * weights) / (weights.sum() + 1e-12)
median_dist = np.median(min_dist)

eval_metrics = {
    'n_candidates': len(candidates),
    'coverage_fraction_demand': float(coverage_fraction),
    'avg_weighted_distance_m': float(avg_weighted_dist),
    'median_distance_m': float(median_dist),
    'num_pois_total': len(poi_ids),
    'total_demand': float(weights.sum())
}

# Save outputs
cand_out = OUTPUT_DIR / f'candidate_stations_P{P}_{METHOD_NAME}.csv'
candidates.to_csv(cand_out, index=False)
pd.Series(eval_metrics).to_frame('value').to_csv(OUTPUT_DIR / f'candidate_eval_P{P}_{METHOD_NAME}.csv')

print('Saved candidates to:', cand_out)
print('Saved evaluation to:', OUTPUT_DIR / f'candidate_eval_P{P}_{METHOD_NAME}.csv')
print('Done.')

#%%
# map_candidates_no_snap_kmeans.py
"""
Create demand-weighted KMeans (NO SNAP) candidate locations (P centroids) and
plot them on an interactive folium map (save HTML). This version DOES NOT snap
centroids to POIs — it uses centroid lon/lat directly. It also handles the
"candidate file needs lat/lon or poi_id" situation by checking the candidate
CSV and joining to POIs when only poi_id is provided.

Usage:
 - Edit the FILE PATHS in USER CONFIG.
 - Run this script in the same environment where pandas, numpy, sklearn and folium are installed.

Outputs:
 - candidate CSV (with centroid_lon/centroid_lat)
 - interactive map HTML saved to OUTPUT_DIR

Note: if you already produced candidates with a separate script, set
CANDIDATE_FILE to that CSV. If it lacks centroid_lon/centroid_lat but has
poi_id, the script will join to the POI file to get coordinates.

"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import folium
from sklearn.neighbors import NearestNeighbors

# ---------------- USER CONFIG ----------------
POI_FILE = r'E:\Uni_PGT\Express\edinburgh_pois.csv'    # original POIs
POI_EDGELIST = r'E:\Uni_PGT\visualisation_outputs\poi_od_sparse_baseline\poi_od_all_clusters_combined_edgelist.csv'
OUTPUT_DIR = Path(r'E:\Uni_PGT\visualisation_outputs\candidate_stations')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If you already have a candidate file from another step, set this; else set to None and script will create centroids
CANDIDATE_FILE = None  # r'E:\path\to\existing_candidates.csv'  # set to file if already produced

P = 300
KM_INIT = 'k-means++'
KMEANS_RANDOM_STATE = 42
# map settings
MAP_CENTER = (55.95, -3.20)   # roughly Edinburgh
MAP_ZOOM = 12

EPS = 1e-9
# ----------------------------------------------

# --- helper: load POIs and compute demand per POI from edgelist ---
print('Loading POIs...')
pois = pd.read_csv(POI_FILE)
pois = pois.dropna(subset=['lat','lon']).reset_index(drop=True)
if 'poi_id' not in pois.columns:
    pois['poi_id'] = pois.index.astype(int)
pois['poi_id'] = pois['poi_id'].astype(int)

# load edgelist to compute demand per POI
print('Loading POI edgelist (this may be large) ...')
edges = pd.read_csv(POI_EDGELIST)
# attempt to infer flow column
cols_lower = {c.lower(): c for c in edges.columns}
flow_col = None
for cand in ('flow', 'demand', 'count', 'weight'):
    if cand in cols_lower:
        flow_col = cols_lower[cand]
        break
if flow_col is None:
    numeric_cols = [c for c in edges.columns if np.issubdtype(edges[c].dtype, np.number)]
    if numeric_cols:
        flow_col = numeric_cols[-1]

if flow_col is None:
    raise RuntimeError('Could not detect numeric flow column in edgelist. Edit script to provide flow column name.')

print('Flow column detected:', flow_col)

inflow = edges.groupby('to_poi_id')[flow_col].sum()
outflow = edges.groupby('from_poi_id')[flow_col].sum()
all_ids = sorted(set(inflow.index.tolist()) | set(outflow.index.tolist()) | set(pois['poi_id'].tolist()))

# demand = inflow + outflow
demand_series = pd.Series(0.0, index=all_ids, dtype=float)
demand_series.loc[inflow.index] += inflow
demand_series.loc[outflow.index] += outflow

poi_demand = pois.set_index('poi_id').reindex(all_ids).copy()
poi_demand['demand'] = demand_series.reindex(poi_demand.index).fillna(0.0)

poi_ids = poi_demand.index.astype(int).tolist()
lats = poi_demand['lat'].to_numpy(dtype=float)
lons = poi_demand['lon'].to_numpy(dtype=float)
weights = poi_demand['demand'].to_numpy(dtype=float)

# avoid zero sample_weight for KMeans
weights_safe = weights.copy()
weights_safe[weights_safe <= 0] = 1e-6

# --- create candidates with NO SNAP (centroids in lon/lat space) ---
if CANDIDATE_FILE is None:
    print(f'Running weighted KMeans (P={P}) to create candidate centroids (NO SNAP) ...')
    # project lon/lat to approximate meters for better geometric clustering
    mean_lat = np.mean(lats)
    xm = (lons - np.mean(lons)) * (111320 * np.cos(np.radians(mean_lat)))
    ym = (lats - np.mean(lats)) * 110540
    X_m = np.column_stack([xm, ym])

    km = KMeans(n_clusters=P, init=KM_INIT, random_state=KMEANS_RANDOM_STATE, n_init=10)
    km.fit(X_m, sample_weight=weights_safe)
    centers_m = km.cluster_centers_

    # map centers back to lon/lat approx
    centers_lon = (centers_m[:,0] / (111320 * np.cos(np.radians(mean_lat)))) + np.mean(lons)
    centers_lat = (centers_m[:,1] / 110540) + np.mean(lats)

    candidates = pd.DataFrame({
        'candidate_id': range(len(centers_lon)),
        'centroid_lon': centers_lon,
        'centroid_lat': centers_lat
    })
    candidate_out = OUTPUT_DIR / f'candidates_kmeans_nosnap_P{P}.csv'
    candidates.to_csv(candidate_out, index=False)
    print('Saved candidate centroids to', candidate_out)
else:
    print('Loading existing candidate file:', CANDIDATE_FILE)
    candidates = pd.read_csv(CANDIDATE_FILE)
    candidate_out = Path(CANDIDATE_FILE)

# --- Ensure we have lon/lat for plotting ---
if 'centroid_lon' in candidates.columns and 'centroid_lat' in candidates.columns:
    centroids_lon = candidates['centroid_lon'].to_numpy(dtype=float)
    centroids_lat = candidates['centroid_lat'].to_numpy(dtype=float)
elif 'lon' in candidates.columns and 'lat' in candidates.columns:
    centroids_lon = candidates['lon'].to_numpy(dtype=float)
    centroids_lat = candidates['lat'].to_numpy(dtype=float)
elif 'poi_id' in candidates.columns:
    # join to POIs to get lat/lon
    print('Candidate file has poi_id but no lon/lat. Joining to POI file to obtain coordinates...')
    join_df = candidates.merge(pois[['poi_id','lat','lon']], left_on='poi_id', right_on='poi_id', how='left')
    if join_df['lat'].isna().any() or join_df['lon'].isna().any():
        missing = join_df[join_df['lat'].isna() | join_df['lon'].isna()]
        raise RuntimeError(f'Some candidate poi_id rows could not be matched to POIs (sample):\n{missing.head().to_string()}')
    centroids_lon = join_df['lon'].to_numpy(dtype=float)
    centroids_lat = join_df['lat'].to_numpy(dtype=float)
    # update candidates with centroid_* columns for consistency
    candidates['centroid_lon'] = centroids_lon
    candidates['centroid_lat'] = centroids_lat
    candidates.to_csv(candidate_out, index=False)
    print('Augmented candidate file with lon/lat and saved to', candidate_out)
else:
    raise RuntimeError('Candidate file needs centroid_lon/centroid_lat or poi_id to join to POIs. Edit candidate CSV.')

# --- Build map with folium ---
print('Building interactive map (folium) ...')
map_center = MAP_CENTER
m = folium.Map(location=map_center, zoom_start=MAP_ZOOM, tiles='cartodbpositron')

# add POI layer (optional: show major POIs as light dots)
# we will sample only POIs with demand > 0 to avoid overcrowding
poi_sample = poi_demand.copy()
poi_sample['demand_norm'] = poi_sample['demand'] / (poi_sample['demand'].max() + EPS)
poi_layer = folium.FeatureGroup(name='POIs (demand)')
for pid, row in poi_sample.iterrows():
    if row['demand'] > 0:
        folium.CircleMarker(location=(row['lat'], row['lon']), radius=2 + 6*row['demand_norm'],
                            popup=f"POI {int(pid)}: {row['demand']:.2f}", fill=True, fill_opacity=0.6).add_to(poi_layer)
# add layer but keep initially off for performance
poi_layer.add_to(m)

# add candidate centroids
cand_layer = folium.FeatureGroup(name=f'Candidates (P={len(centroids_lon)})')
for cid, lon, lat in zip(candidates.get('candidate_id', range(len(centroids_lon))), centroids_lon, centroids_lat):
    folium.CircleMarker(location=(lat, lon), radius=6, color='red', fill=True, fill_color='red',
                        popup=f'candidate {cid}').add_to(cand_layer)
cand_layer.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

map_out = OUTPUT_DIR / f'candidates_map_kmeans_nosnap_P{P}.html'
m.save(str(map_out))
print('Saved interactive map to', map_out)

print('Done.')
