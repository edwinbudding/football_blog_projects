import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Load your dataset
file_path = '/Users/anokhpalakurthi/Downloads/Every NFL Draft from 2000 to 2020.xlsx' 
data = pd.read_excel(file_path)

# Ensure correct data types
data["Pick"] = pd.to_numeric(data["Pick"], errors='coerce')
data["DrAV"] = pd.to_numeric(data["DrAV"], errors='coerce')

# Filter data to include only valid picks and DrAV
data = data.dropna(subset=["Pick", "DrAV"])

# Group by the "Pick" column and calculate the mean DrAV for each pick
expected_drav = data.groupby("Pick")["DrAV"].mean().reset_index()
expected_drav.columns = ["Pick", "Expected_DrAV"]
expected_drav["Expected_DrAV"] = expected_drav["Expected_DrAV"].round(2)

# Calculate the median DrAV for each pick
median_drav = data.groupby("Pick")["DrAV"].median().reset_index()
median_drav.columns = ["Pick", "Median_DrAV"]

# Remove outliers using IQR for each pick
def remove_outliers(group):
    q1 = group["DrAV"].quantile(0.25)
    q3 = group["DrAV"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return group[(group["DrAV"] >= lower_bound) & (group["DrAV"] <= upper_bound)]

# Apply the function and reset the index
filtered_data = data.groupby("Pick").apply(remove_outliers).reset_index(drop=True)

# Calculate filtered mean DrAV
filtered_mean_drav = filtered_data.groupby("Pick")["DrAV"].mean().reset_index()
filtered_mean_drav.columns = ["Pick", "Filtered_Expected_DrAV"]

# Weighted DrAV calculation
weighted_drav = data.groupby("Pick").apply(
    lambda x: (x["DrAV"].sum()) / len(x)
).reset_index(name="Weighted_Expected_DrAV")

# Combine into a summary table with additional stats
summary_stats = data.groupby("Pick")["DrAV"].agg(["mean", "median", "std"]).reset_index()
summary_stats.columns = ["Pick", "Mean_DrAV", "Median_DrAV", "Std_Dev"]
summary_stats["Mean_DrAV"] = summary_stats["Mean_DrAV"].round(2)
summary_stats["Median_DrAV"] = summary_stats["Median_DrAV"].round(2)
summary_stats["Std_Dev"] = summary_stats["Std_Dev"].round(2)

# Define Heuristic Ranges
ranges = pd.cut(
    expected_drav["Pick"],
    bins=[1, 5, 15, 32, 64, 100, 224],  # Adjust ranges as needed
    labels=["Top 1", "2-5", "6-15", "16-32", "33-64", "65-224"],
    include_lowest=True
)

expected_drav["Pick_Range"] = ranges

# Calculate summary stats for ranges
range_summary = expected_drav.groupby("Pick_Range").agg(
    {"Expected_DrAV": ["mean", "std"], "Pick": "count"}
).reset_index()

# Flatten MultiIndex columns
range_summary.columns = ["Pick_Range", "Mean_DrAV", "Std_Dev", "Count"]

# Fit a polynomial regression to smooth the Expected DrAV trend
x = expected_drav["Pick"]
y = expected_drav["Expected_DrAV"]
trend = Polynomial.fit(x, y, 3)
x_smooth = np.linspace(x.min(), x.max(), 500)
y_smooth = trend(x_smooth)

# Plot the trendline
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Mean DrAV", marker="o", linestyle="", alpha=0.6)
plt.plot(x_smooth, y_smooth, label="Expected DrAV Trendline", color="red")
plt.xlabel("Draft Pick")
plt.ylabel("Expected DrAV")
plt.title("Expected DrAV Trendline by Pick")
plt.legend()
plt.grid()
plt.show()

# Save the polynomial trendline values into the DataFrame
trend_values = trend(expected_drav["Pick"])
expected_drav["Trendline_DrAV"] = trend_values.round(2)

# Save all results to Excel
excel_output_file = '/Users/anokhpalakurthi/Downloads/draft_analysis_results_with_trendline.xlsx'

# Open a writer context and save all DataFrames to Excel
with pd.ExcelWriter(excel_output_file, engine='openpyxl') as writer:
    expected_drav.to_excel(writer, sheet_name="Mean DrAV", index=False)
    median_drav.to_excel(writer, sheet_name="Median DrAV", index=False)
    filtered_mean_drav.to_excel(writer, sheet_name="Filtered Mean DrAV", index=False)
    weighted_drav.to_excel(writer, sheet_name="Weighted DrAV", index=False)
    summary_stats.to_excel(writer, sheet_name="Summary Stats", index=False)
    range_summary.to_excel(writer, sheet_name="Range Summary", index=False)

print("Analysis complete! Results saved to:", excel_output_file)