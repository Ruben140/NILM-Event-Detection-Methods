import numpy as np
import pandas as pd
from scipy.signal import medfilt
# from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def local_threshold_event_detection(power_signal, window_size=60, a=7, b=6, min_threshold=200):
    # Step 2: Apply median filter
    median_filtered_signal = medfilt(power_signal)

    # Step 3-6: Compute Δ𝑃
    delta_p = np.abs(np.diff(median_filtered_signal))

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

    #Step 15-17: Compute statistics
    sp = []
    sa = []
    wp = []
    for i in range(len(delta_power)-1):
        sp.append(np.mean(mean_list[i]) + np.mean(std_list[i]))
        sa.append(sp[i]/2)
        if(std_list[i] > sa[i]):
            wp.append(i)

    # Step 18-25: Peak detection
    event_indices = []
    for i in wp:
        # Perform peak detection
        peaks, _ = find_peaks(delta_p[i:i + window_size])

        # Calculate threshold
        threshold = a * mean_list[i] + b * std_list[i]

        # # Confirm whether the threshold is greater than the minimum threshold
        if threshold < min_threshold:
            threshold = min_threshold

        # Filter peaks based on threshold
        selected_peaks = peaks[delta_p[i + peaks] > threshold]

        # Map event indices back to the original signal
        event_indices.extend([(i + w) for w in selected_peaks])

    # Filter for duplicates in the event_indices
    event_indices = list(set(event_indices))

    return event_indices

# Read the CSV file with Saxion data
df = pd.read_csv()

# Convert the 'timestamp' column to datetime format
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='ISO8601')

# Round the timestamp to the nearest 200ms
# This is because the data per sample can be spread over multiple entries. (Only goes for Saxion data.)
# However, the time between samples is >200ms and therefore we can group them by the nearest 200ms.
df['ROUNDED_TIMESTAMP'] = df['TIMESTAMP'].dt.round('200ms')

# Retrieve the power signal column from the data.
result = df.groupby('ROUNDED_TIMESTAMP').agg({
    'POW': 'last',
})

# Execute LocalThreshold-algorithm on the power signal
event_indices = local_threshold_event_detection(result['POW'].values)

# Create a new column to store the event indices
result['Event_Index'] = np.nan

# Update the event indices column based on the detected events
result.loc[result.index[event_indices], 'Event_Index'] = event_indices

# Plot the power aggregation signal with detected events
plt.figure(figsize=(10, 6))

# Plot the power aggregation signal
plt.plot(result.index, result['POW'], label='Power Aggregation Signal', color='blue')

# Highlight the detected events
plt.scatter(result.index[result['Event_Index'].notna()], result['POW'][result['Event_Index'].notna()], label='Detected Events', s=50, color='red')

# Add labels and legend
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Power Aggregation Signal with Local Threshold-based Event Detection')
plt.legend()
plt.show()

# To print the events as (timestamp, value) use the following print statement
# print(result.index[result['Event_Index'].notna()].strftime('%H:%M:%S.%f').tolist())
