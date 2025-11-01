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
OUTPUT_FILE = Path(r'E:\Uni_PGT\counts-data\combined_od_with_date.csv')
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
