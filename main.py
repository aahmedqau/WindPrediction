import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Step 1: Load Data ---
file_path = 'Copy of Dataset.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

# --- Step 2: Parse Data ---
times = []
true_radiation = []
predicted_radiation = []

i = 0
while i < len(df):
    try:
        # Try to parse timestamp
        time_stamp = df.iloc[i, 0]
        if isinstance(time_stamp, str) and ':' in time_stamp:
            times.append(time_stamp)
            true_radiation.append(float(df.iloc[i+1, 0]))
            predicted_radiation.append(float(df.iloc[i+2, 0]))
            i += 6  # Skip to next block
        else:
            i += 1
    except Exception as e:
        i += 1

# --- Step 3: Generate Hour-Based Time Axis ---
# Convert to hour-based timeline assuming hourly data
hours = np.arange(len(true_radiation))

# Convert lists to numpy arrays
true_radiation = np.array(true_radiation)
predicted_radiation = np.array(predicted_radiation)

# --- Step 4: Define Zoom Periods ---
zoom_periods = [
    {'start': 0, 'end': 24, 'title': 'Zoomed-In: Day 1 Diurnal Pattern'},
    {'start': 720, 'end': 744, 'title': 'Zoomed-In: Midpoint Fluctuation'},
    {'start': 840, 'end': 864, 'title': 'Zoomed-In: End of Series'}
]

# --- Step 5: Metrics Calculation Function ---
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return f"RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}"

# --- Step 6: Create Figure Layout ---
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 12), sharex=False)

# --- Main Plot ---
ax_main = axes[0]
ax_main.plot(hours, true_radiation, label='True Solar', color='blue', marker='o', markevery=24, markersize=4)
ax_main.plot(hours, predicted_radiation, label='Predicted Solar', color='orange', linestyle='--', marker='s', markevery=24, markersize=4)
ax_main.set_title('Radiation Intensity: True vs. Predicted (Full Time Series)', fontsize=14)
ax_main.set_ylabel('Radiation (W/m²)')
ax_main.grid(True, linestyle='--', alpha=0.5)
ax_main.legend(loc='upper right')

# Highlight zoom regions
for i, period in enumerate(zoom_periods):
    ax_main.axvspan(period['start'], period['end'], color=f'C{i+1}', alpha=0.1)

# --- Zoom Plots ---
for i, (ax_zoom, period) in enumerate(zip(axes[1:], zoom_periods)):
    start, end = period['start'], period['end']
    x_range = hours[start:end]
    y_true = true_radiation[start:end]
    y_pred = predicted_radiation[start:end]

    ax_zoom.plot(x_range, y_true, label='True Solar', color='blue', marker='o', markevery=2, markersize=4)
    ax_zoom.plot(x_range, y_pred, label='Predicted Solar', color='orange', linestyle='--', marker='s', markevery=2, markersize=4)

    ax_zoom.set_title(period['title'])
    ax_zoom.set_xlabel('Time (h)')
    ax_zoom.set_ylabel('Radiation (W/m²)')
    ax_zoom.grid(True, linestyle='--', alpha=0.5)

    # Annotate metrics
    metrics = compute_metrics(y_true, y_pred)
    ax_zoom.annotate(metrics, xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()
