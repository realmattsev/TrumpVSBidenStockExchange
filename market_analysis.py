"""
Complete market drop analysis comparing Trump and Biden presidential periods
- TRUMP: January 20, 2017 to January 19, 2021
- BIDEN: January 20, 2021 to January 19, 2025

Features:
1. Downloads S&P 500 data using yfinance
2. Filters by presidential time windows
3. Categorizes daily drops into 3 tiers
4. Visualizes drop frequency comparison
5. Calculates recovery time metrics
6. Generates summary table and charts
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("Set1")

# Create output directory for charts if it doesn't exist
if not os.path.exists('charts'):
    os.makedirs('charts')

# Step 1: Download S&P 500 data
print("Downloading S&P 500 data...")
ticker = "^GSPC"
df = yf.download(ticker, start="2017-01-01", end="2025-01-20")

# Calculate daily percentage changes
df["Pct_Change"] = df["Close"].pct_change() * 100

# Step 2: Define time windows
print("Filtering data by presidential periods...")
trump_period = df.loc["2017-01-20":"2021-01-19"].copy()
biden_period = df.loc["2021-01-20":datetime.now().strftime('%Y-%m-%d')].copy()

# Step 3: Define drop categorization
TIERS = {
    "Tier 1 (1%-2%)": (-2, -1),
    "Tier 2 (2%-4%)": (-4, -2),
    "Tier 3 (>4%)": (-100, -4)
}

def categorize_drops(df, tiers):
    """Count market drops in different severity tiers"""
    results = {}
    for label, (low, high) in tiers.items():
        mask = (df["Pct_Change"] <= high) & (df["Pct_Change"] > low)
        results[label] = df[mask].shape[0]
    
    # Also include trading days count for normalization
    results["Trading Days"] = len(df)
    return results

def analyze_recovery(df, drop_threshold=-4):
    """Calculate days to recover from significant drops"""
    recovery_days = []
    drop_dates = []
    drop_values = []
    recovery_dates = []
    
    prices = df["Close"]
    for i in range(1, len(df)-1):
        if df["Pct_Change"].iloc[i] <= drop_threshold:
            drop_day = df.index[i]
            drop_price = prices.iloc[i]
            drop_values.append(df["Pct_Change"].iloc[i])
            drop_dates.append(drop_day)
            
            # Find recovery day (when price returns to or exceeds the pre-drop level)
            recovered = False
            for j in range(i+1, len(df)):
                if prices.iloc[j] >= drop_price:
                    recovery_day = df.index[j]
                    days_to_recover = (recovery_day - drop_day).days
                    recovery_days.append(days_to_recover)
                    recovery_dates.append(recovery_day)
                    recovered = True
                    break
                    
            # If not recovered by end of period
            if not recovered:
                recovery_days.append(np.nan)
                recovery_dates.append(None)
    
    # Create summary dataframe of drops
    if drop_dates:
        recovery_df = pd.DataFrame({
            'Drop Date': drop_dates,
            'Drop %': drop_values,
            'Recovery Date': recovery_dates,
            'Days to Recover': recovery_days
        })
        return recovery_df
    else:
        return pd.DataFrame()

# Step 4: Run the analysis
print("Analyzing market drops...")
trump_drops = categorize_drops(trump_period, TIERS)
biden_drops = categorize_drops(biden_period, TIERS)

# Calculate volatility metrics
print("Calculating volatility metrics...")
# Calculate 30-day rolling volatility (standard deviation of daily returns)
trump_volatility = trump_period["Pct_Change"].rolling(window=30).std()
biden_volatility = biden_period["Pct_Change"].rolling(window=30).std()

# Average volatility
trump_avg_volatility = trump_volatility.mean()
biden_avg_volatility = biden_volatility.mean()

# Calculate drops per year for normalization (accounting for different period lengths)
trump_years = (trump_period.index[-1] - trump_period.index[0]).days / 365.25
biden_years = (biden_period.index[-1] - biden_period.index[0]).days / 365.25

# For recovery analysis
trump_recovery = analyze_recovery(trump_period, -4)
biden_recovery = analyze_recovery(biden_period, -4)

# Step 5: Create summary tables
print("Generating summary tables...")
# Raw drop counts
summary_counts = pd.DataFrame([trump_drops, biden_drops], 
                     index=["TRUMP (2017-2021)", "BIDEN (2021-Present)"])

# Annualized drop metrics (drops per year)
trump_annual = {k: v/trump_years if k != "Trading Days" else v for k, v in trump_drops.items()}
biden_annual = {k: v/biden_years if k != "Trading Days" else v for k, v in biden_drops.items()}
summary_annual = pd.DataFrame([trump_annual, biden_annual], 
                     index=["TRUMP (annualized)", "BIDEN (annualized)"])

# Step 6: Print summary statistics
print("\n--- MARKET DROP FREQUENCY ---")
print(summary_counts)

print("\n--- ANNUALIZED DROP METRICS ---")
print(summary_annual)

print("\n--- RECOVERY TIME ANALYSIS ---")
if not trump_recovery.empty:
    # Exclude NaN values for mean calculation (drops that never recovered)
    trump_recovery_days = trump_recovery['Days to Recover'].dropna()
    unrecovered_count = trump_recovery['Days to Recover'].isna().sum()
    
    if len(trump_recovery_days) > 0:
        print(f"TRUMP: {len(trump_recovery)} major drops (>4%)")
        print(f"      Average recovery time: {trump_recovery_days.mean():.1f} days")
        print(f"      Longest recovery: {trump_recovery_days.max():.0f} days")
        if unrecovered_count > 0:
            print(f"      {unrecovered_count} drops never recovered by end of term")
    else:
        print(f"TRUMP: {len(trump_recovery)} major drops (>4%), none recovered by end of term")
else:
    print("TRUMP: No large drops (>4%) requiring recovery analysis")
    
if not biden_recovery.empty:
    # Exclude NaN values for mean calculation (drops that never recovered)
    biden_recovery_days = biden_recovery['Days to Recover'].dropna()
    unrecovered_count = biden_recovery['Days to Recover'].isna().sum()
    
    if len(biden_recovery_days) > 0:
        print(f"BIDEN: {len(biden_recovery)} major drops (>4%)")
        print(f"      Average recovery time: {biden_recovery_days.mean():.1f} days")
        print(f"      Longest recovery: {biden_recovery_days.max():.0f} days")
        if unrecovered_count > 0:
            print(f"      {unrecovered_count} drops never recovered by present day")
    else:
        print(f"BIDEN: {len(biden_recovery)} major drops (>4%), none recovered by present day")
else:
    print("BIDEN: No large drops (>4%) requiring recovery analysis")

# Step 7: Visualizations
print("\nGenerating visualizations...")
# 1. Drops comparison chart
plt.figure(figsize=(12, 6))
summary_counts.iloc[:, :-1].T.plot(kind='bar', color=['red', 'blue'])
plt.title('Market Drop Comparison: Trump vs Biden', fontsize=16)
plt.ylabel('Number of Days', fontsize=14)
plt.xlabel('Drop Severity', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('charts/drops_comparison.png')

# 2. Normalized drops per year
plt.figure(figsize=(12, 6))
summary_annual.iloc[:, :-1].T.plot(kind='bar', color=['darkred', 'darkblue'])
plt.title('Annualized Market Drops: Trump vs Biden', fontsize=16)
plt.ylabel('Drops per Year', fontsize=14)
plt.xlabel('Drop Severity', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('charts/drops_annual.png')

# 3. Price charts with major drops highlighted
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(trump_period.index, trump_period['Close'], color='red', linewidth=1)
if not trump_recovery.empty:
    plt.scatter(trump_recovery['Drop Date'], 
                trump_period.loc[trump_recovery['Drop Date']]['Close'],
                color='black', s=50, marker='v')
plt.title('S&P 500 during Trump Presidency with Major Drops (>4%)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(biden_period.index, biden_period['Close'], color='blue', linewidth=1)
if not biden_recovery.empty:
    plt.scatter(biden_recovery['Drop Date'], 
                biden_period.loc[biden_recovery['Drop Date']]['Close'],
                color='black', s=50, marker='v')
plt.title('S&P 500 during Biden Presidency with Major Drops (>4%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/price_charts.png')

# 4. Recovery time histogram (if we have recovery data)
if not trump_recovery.empty and not biden_recovery.empty:
    plt.figure(figsize=(12, 6))
    
    # Use only finite values for histogram
    trump_days = trump_recovery['Days to Recover'].dropna().values
    biden_days = biden_recovery['Days to Recover'].dropna().values
    
    if len(trump_days) > 0 and len(biden_days) > 0:
        bins = max(5, min(10, max(len(trump_days), len(biden_days))))
        plt.hist([trump_days, biden_days], bins=bins,
                color=['red', 'blue'], alpha=0.7, label=['Trump', 'Biden'])
        plt.title('Recovery Time from Large Drops (>4%)', fontsize=16)
        plt.xlabel('Days to Recover', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('charts/recovery_histogram.png')

# 5. NEW: Normalized performance comparison 
# This aligns both administrations to a common starting point (100) for direct comparison
plt.figure(figsize=(12, 6))
trump_norm = trump_period['Close'] / trump_period['Close'].iloc[0] * 100
biden_norm = biden_period['Close'] / biden_period['Close'].iloc[0] * 100

# Create day indices (0, 1, 2, ...) for x-axis
trump_days = np.arange(len(trump_norm))
biden_days = np.arange(len(biden_norm))

plt.plot(trump_days, trump_norm, 'r-', linewidth=2, label='Trump')
plt.plot(biden_days, biden_norm, 'b-', linewidth=2, label='Biden')

# Set the x-axis to show years in office
max_days = max(len(trump_days), len(biden_days))
plt.xticks([0, 252, 504, 756, 1008], ['Year 0', 'Year 1', 'Year 2', 'Year 3', 'Year 4'])

plt.grid(True, alpha=0.3)
plt.title('Normalized S&P 500 Performance Comparison\n(Starting Point = 100)', fontsize=16)
plt.ylabel('Normalized Price', fontsize=14)
plt.xlabel('Time in Office', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('charts/normalized_comparison.png')

# 6. Volatility comparison chart
plt.figure(figsize=(12, 6))

# Create subplot for Trump volatility
plt.subplot(2, 1, 1)
plt.plot(trump_period.index, trump_volatility, 'r-', linewidth=1)
plt.axhline(y=trump_avg_volatility, color='darkred', linestyle='--', 
           label=f'Avg: {trump_avg_volatility:.2f}%')
plt.title('30-Day Rolling Volatility: Trump Administration', fontsize=14)
plt.ylabel('Volatility (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Create subplot for Biden volatility  
plt.subplot(2, 1, 2)
plt.plot(biden_period.index, biden_volatility, 'b-', linewidth=1)
plt.axhline(y=biden_avg_volatility, color='darkblue', linestyle='--',
           label=f'Avg: {biden_avg_volatility:.2f}%')
plt.title('30-Day Rolling Volatility: Biden Administration', fontsize=14)
plt.ylabel('Volatility (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('charts/volatility_comparison.png')

# Print volatility summary
print("\n--- VOLATILITY COMPARISON ---")
print(f"TRUMP: Average 30-day volatility: {trump_avg_volatility:.2f}%")
print(f"BIDEN: Average 30-day volatility: {biden_avg_volatility:.2f}%")
print(f"Difference: {(biden_avg_volatility - trump_avg_volatility):.2f}%")

print("\nAnalysis complete! Visualization files saved to the 'charts' directory.") 