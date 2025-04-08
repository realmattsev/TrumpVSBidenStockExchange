"""
Enhanced market drop analysis script comparing two US presidential periods:
- TRUMP: Jan 20, 2017 to Jan 19, 2021
- BIDEN: Jan 20, 2021 to Jan 19, 2025

Features:
1. Downloads S&P 500 data using yfinance
2. Filters by time window
3. Categorizes daily drops into tiers
4. Visualizes drop frequency comparison
5. Analyzes recovery time from large drops
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Download data
ticker = "^GSPC"
df = yf.download(ticker, start="2017-01-01", end="2025-01-20")

# Print dataframe columns to debug
print("DataFrame columns:", df.columns.tolist())

# Use 'Close' instead of 'Adj Close' as column names may have changed
df["Pct Change"] = df["Close"].pct_change() * 100

# Step 2: Define time windows
trump_period = df.loc["2017-01-20":"2021-01-19"].copy()
biden_period = df.loc["2021-01-20":"2025-01-19"].copy()

# Step 3: Drop categorization function with flexible tiers
DROP_TIERS = {
    "Tier 1 (1%-2%)": (-2, -1),
    "Tier 2 (2%-4%)": (-4, -2),
    "Tier 3 (4%-6%)": (-6, -4),
    "Tier 4 (>6%)": (-100, -6),  # For extreme drops
}

def categorize_drops(df, tiers):
    drops = df[df["Pct Change"] < -1]
    results = {}
    for label, (low, high) in tiers.items():
        results[label] = drops[(drops["Pct Change"] <= high) & (drops["Pct Change"] > low)].shape[0]
    return results

# Step 4: Recovery time analysis
def analyze_recovery(df, drop_threshold=-4):
    recovery_days = []
    prices = df["Close"]  # Use 'Close' instead of 'Adj Close'
    for i in range(1, len(df)-1):
        if df["Pct Change"].iloc[i] <= drop_threshold:
            drop_day = df.index[i]
            drop_price = prices.iloc[i]
            # Find the day it recovers back to or above the drop day price
            for j in range(i+1, len(df)):
                if prices.iloc[j] >= drop_price:
                    recovery_day = df.index[j]
                    days_to_recover = (recovery_day - drop_day).days
                    recovery_days.append(days_to_recover)
                    break
    return recovery_days

# Step 5: Run the analysis
trump_drops = categorize_drops(trump_period, DROP_TIERS)
biden_drops = categorize_drops(biden_period, DROP_TIERS)

# For recovery analysis
trump_recovery = analyze_recovery(trump_period)
biden_recovery = analyze_recovery(biden_period)

# Step 6: Create summary dataframe
summary = pd.DataFrame([trump_drops, biden_drops], 
                      index=["TRUMP (2017-2021)", "BIDEN (2021-2025)"])

# Print summary statistics
print("\n--- DROP FREQUENCY COMPARISON ---")
print(summary)

print("\n--- RECOVERY TIME ANALYSIS ---")
if trump_recovery:
    print(f"TRUMP: Average recovery days for large drops: {np.mean(trump_recovery):.1f}")
else:
    print("TRUMP: No large drops requiring recovery analysis")
    
if biden_recovery:
    print(f"BIDEN: Average recovery days for large drops: {np.mean(biden_recovery):.1f}")
else:
    print("BIDEN: No large drops requiring recovery analysis")

# Step 7: Visualizations
# Drops comparison chart
plt.figure(figsize=(12, 6))
summary.T.plot(kind='bar', color=['red', 'blue'])
plt.title('Market Drop Comparison: Trump vs Biden')
plt.ylabel('Number of Days')
plt.xlabel('Drop Severity')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('drops_comparison.png')

# Recovery time histogram (if there are recovery days)
if trump_recovery and biden_recovery:
    plt.figure(figsize=(12, 6))
    plt.hist([trump_recovery, biden_recovery], bins=10, 
             color=['red', 'blue'], alpha=0.7, label=['Trump', 'Biden'])
    plt.title('Recovery Time from Large Drops (>4%)')
    plt.xlabel('Days to Recover')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('recovery_comparison.png')

print("\nAnalysis complete! Visualization files saved.")
