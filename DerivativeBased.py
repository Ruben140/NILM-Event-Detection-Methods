import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np
from scipy.signal import find_peaks
import pandas as pd

def preprocess_power_signal(power, Vnorm=220):
    power_filtered = medfilt(power)

    Irms = np.sqrt(power_filtered / Vnorm)
    return Irms

def changes_detection(Irms):
    dIrms = np.diff(Irms)
    ps = np.square(dIrms)
    ps = np.where(dIrms < 0, -ps, ps)
    return ps

def peak_detection(ps, Wd, theta_th):
    rising_edges, _ = find_peaks(ps, distance=Wd, height=theta_th)
    falling_edges, _ = find_peaks(-ps, distance=Wd, height=theta_th)
    tau = np.sort(np.concatenate([rising_edges, falling_edges]))
    return tau

# Read the CSV file
df = pd.read_csv()

# Convert the 'timestamp' column to datetime format
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='ISO8601')

# Round the timestamp to the nearest 200ms, as the data is sampled in <200ms intervals
df['ROUNDED_TIMESTAMP'] = df['TIMESTAMP'].dt.round('200ms')

# Step 1: Input is the power consumption signal
result = df.groupby('ROUNDED_TIMESTAMP').agg({
    'POW': 'last',
})

# Remove all NaN values
result = result.dropna(subset=['POW'])

# Step 1: Preprocessing
Irms = preprocess_power_signal(result['POW'].values)

# Step 2: Changes detection
ps = changes_detection(Irms)

# Step 3: Peak detector
distance = 1250  # Minimum distance between peaks
theta = 0.45  # Threshold to filter out peaks
tau = peak_detection(ps, distance, theta)

# Rename for ease of use
event_indices = tau

# Create a new column to store the event indices
result['EVENTS'] = np.nan

# Update the event indices column based on the detected events
result.loc[result.index[event_indices], 'EVENTS'] = event_indices

# print(result.index[result['EVENTS'].notna()].strftime('%H:%M:%S.%f').tolist())

# Plot the power aggregation signal with detected events
plt.figure(figsize=(10, 6))

# Plot the power aggregation signal
plt.plot(result.index, result['POW'], label='Power Aggregation Signal', color='blue')

# Highlight the detected events
plt.scatter(result.index[result['EVENTS'].notna()], result['POW'][result['EVENTS'].notna()], label='Detected Events (Adaptive)', s=50, color='green')

# Add labels and legend
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Power Aggregation Signal with derivative-based Event Detection')
plt.legend()
plt.show()
