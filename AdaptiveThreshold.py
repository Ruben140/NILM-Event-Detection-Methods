import numpy as np
import pandas as pd
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def adaptive_threshold_event_detection(power_signal, window_size, t1, q):
    # Step 2: Apply median filter
    median_filtered_signal = medfilt(power_signal)

    # Step 3-6: Compute Δ𝑃
    delta_p = np.abs(np.diff(median_filtered_signal))
    # delta_p = np.diff(median_filtered_signal)

    # Step 8-10: Compute ΔPower
    delta_power = []
    for i in range(len(delta_p) - window_size + 1):
        delta_power.append(delta_p[i:i+window_size])

    # Step 11-14: Compute mean and standard deviation for each window
    mean_list = []
    std_list = []
    for i in delta_power:
        mean_list.append(np.mean(i))
        std_list.append(np.std(i))

    # Convert mu_list and sigma_list to NumPy arrays
    mean_array = np.array(mean_list)
    std_array = np.array(std_list)

    # Calculate t and r
    t = window_size
    r = np.max(power_signal) / np.mean(power_signal)

    # Calculate threshold value
    threshold = (((t / np.sqrt(r)) * std_array) + (t1 * mean_array)) * q

    # Step 18-25: Peak detection using adaptive threshold
    event_indices = []
    for i, threshold_value in enumerate(threshold):
        # Perform peak detection
        peaks, _ = find_peaks(delta_p[i:i + window_size])

        # Assuming delta_power, threshold_value, and peaks are defined
        # Check if delta_power is bigger than the threshold and 250ß
        selected_peaks = peaks[(delta_p[i + peaks] > threshold_value) & (delta_p[i + peaks] > 250)]

        # Map event indices back to the original signal
        event_indices.extend([(i + w) for w in selected_peaks])

    return event_indices

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

# Example usage with adjusted parameters
event_indices = adaptive_threshold_event_detection(result['POW'].values, window_size=60, t1=1, q=0.4)

# Create a new column to store the event indices
result['Event_Index_Adaptive'] = np.nan

# Update the event indices column based on the detected events
result.loc[result.index[event_indices], 'Event_Index_Adaptive'] = event_indices

print(result.index[result['Event_Index_Adaptive'].notna()].strftime('%H:%M:%S.%f').tolist())

# Plot the power aggregation signal with detected events
plt.figure(figsize=(10, 6))

# Plot the power aggregation signal
plt.plot(result.index, result['POW'], label='Power Aggregation Signal', color='blue')

# Highlight the detected events
plt.scatter(result.index[result['Event_Index_Adaptive'].notna()], result['POW'][result['Event_Index_Adaptive'].notna()], label='Detected Events (Adaptive)', s=50, color='green')

# Add labels and legend
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Power Aggregation Signal with Adaptive Threshold-based Event Detection')
plt.legend()
plt.show()
