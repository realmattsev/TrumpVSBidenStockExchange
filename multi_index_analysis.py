"""
Comprehensive market analysis comparing multiple indices during Trump and Biden presidencies
- TRUMP: January 20, 2017 to January 19, 2021
- BIDEN: January 20, 2021 to present

Features:
1. Analyzes S&P 500, Nasdaq, and Dow Jones Industrial Average
2. Filters by presidential time windows
3. Categorizes daily drops into tiers
4. Calculates recovery time metrics
5. Measures volatility for each index and period
6. Generates comparative visualizations
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
if not os.path.exists('charts_multi'):
    os.makedirs('charts_multi')

# Parameters
DROP_THRESHOLD = -4  # Threshold for major drops
TIERS = {
    "Tier 1 (1%-2%)": (-2, -1),
    "Tier 2 (2%-4%)": (-4, -2),
    "Tier 3 (>4%)": (-100, -4)
}

# Track indices to analyze
INDICES = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI"
}

class MarketAnalyzer:
    def __init__(self, index_name, ticker_symbol):
        self.index_name = index_name
        self.ticker = ticker_symbol
        self.df = None
        self.trump_period = None
        self.biden_period = None
        self.trump_drops = None
        self.biden_drops = None
        self.trump_recovery = None
        self.biden_recovery = None
        self.trump_volatility = None
        self.biden_volatility = None
        self.trump_volatility_avg = None
        self.biden_volatility_avg = None
        self.summary_counts = None
        self.summary_annual = None
        
    def download_data(self):
        """Download historical data for the index"""
        print(f"Downloading {self.index_name} data...")
        self.df = yf.download(self.ticker, start="2017-01-01", end="2025-01-20")
        self.df["Pct_Change"] = self.df["Close"].pct_change() * 100
        
    def filter_periods(self):
        """Filter data into presidential periods"""
        print(f"Filtering {self.index_name} data by presidential periods...")
        self.trump_period = self.df.loc["2017-01-20":"2021-01-19"].copy()
        self.biden_period = self.df.loc["2021-01-20":datetime.now().strftime('%Y-%m-%d')].copy()
        
    def analyze_drops(self):
        """Analyze drop frequencies and metrics"""
        print(f"Analyzing {self.index_name} market drops...")
        self.trump_drops = self.categorize_drops(self.trump_period, TIERS)
        self.biden_drops = self.categorize_drops(self.biden_period, TIERS)
        
        # Calculate volatility metrics
        self.trump_volatility = self.trump_period["Pct_Change"].rolling(window=30).std()
        self.biden_volatility = self.biden_period["Pct_Change"].rolling(window=30).std()
        self.trump_volatility_avg = self.trump_volatility.mean()
        self.biden_volatility_avg = self.biden_volatility.mean()
        
        # For recovery analysis
        self.trump_recovery = self.analyze_recovery(self.trump_period, DROP_THRESHOLD)
        self.biden_recovery = self.analyze_recovery(self.biden_period, DROP_THRESHOLD)
        
        # Calculate drops per year (to account for different period lengths)
        trump_years = (self.trump_period.index[-1] - self.trump_period.index[0]).days / 365.25
        biden_years = (self.biden_period.index[-1] - self.biden_period.index[0]).days / 365.25
        
        # Create summary tables
        self.summary_counts = pd.DataFrame([self.trump_drops, self.biden_drops], 
                             index=[f"TRUMP (2017-2021)", f"BIDEN (2021-Present)"])
        
        # Annualized drop metrics
        trump_annual = {k: v/trump_years if k != "Trading Days" else v for k, v in self.trump_drops.items()}
        biden_annual = {k: v/biden_years if k != "Trading Days" else v for k, v in self.biden_drops.items()}
        self.summary_annual = pd.DataFrame([trump_annual, biden_annual], 
                             index=[f"TRUMP (annualized)", f"BIDEN (annualized)"])
    
    def categorize_drops(self, df, tiers):
        """Count market drops in different severity tiers"""
        results = {}
        for label, (low, high) in tiers.items():
            mask = (df["Pct_Change"] <= high) & (df["Pct_Change"] > low)
            results[label] = df[mask].shape[0]
        
        # Also include trading days count for normalization
        results["Trading Days"] = len(df)
        return results
    
    def analyze_recovery(self, df, drop_threshold):
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
                if isinstance(drop_price, pd.Series):
                    drop_price = drop_price.iloc[0]
                
                drop_values.append(df["Pct_Change"].iloc[i])
                drop_dates.append(drop_day)
                
                # Find recovery day (when price returns to or exceeds pre-drop level)
                recovered = False
                for j in range(i+1, len(df)):
                    current_price = prices.iloc[j]
                    if isinstance(current_price, pd.Series):
                        current_price = current_price.iloc[0]
                    
                    if current_price >= drop_price:
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
    
    def print_summary(self):
        """Print summary statistics for this index"""
        print(f"\n=== {self.index_name} ANALYSIS ===")
        
        print("\n--- DROP FREQUENCY ---")
        print(self.summary_counts)
        
        print("\n--- ANNUALIZED DROP METRICS ---")
        print(self.summary_annual)
        
        print("\n--- RECOVERY TIME ANALYSIS ---")
        if not self.trump_recovery.empty:
            # Exclude NaN values for mean calculation
            trump_recovery_days = self.trump_recovery['Days to Recover'].dropna()
            unrecovered_count = self.trump_recovery['Days to Recover'].isna().sum()
            
            if len(trump_recovery_days) > 0:
                print(f"TRUMP: {len(self.trump_recovery)} major drops (>4%)")
                print(f"      Average recovery time: {trump_recovery_days.mean():.1f} days")
                print(f"      Longest recovery: {trump_recovery_days.max():.0f} days")
                if unrecovered_count > 0:
                    print(f"      {unrecovered_count} drops never recovered by end of term")
            else:
                print(f"TRUMP: {len(self.trump_recovery)} major drops (>4%), none recovered by end of term")
        else:
            print("TRUMP: No large drops (>4%) requiring recovery analysis")
            
        if not self.biden_recovery.empty:
            # Exclude NaN values for mean calculation
            biden_recovery_days = self.biden_recovery['Days to Recover'].dropna()
            unrecovered_count = self.biden_recovery['Days to Recover'].isna().sum()
            
            if len(biden_recovery_days) > 0:
                print(f"BIDEN: {len(self.biden_recovery)} major drops (>4%)")
                print(f"      Average recovery time: {biden_recovery_days.mean():.1f} days")
                print(f"      Longest recovery: {biden_recovery_days.max():.0f} days")
                if unrecovered_count > 0:
                    print(f"      {unrecovered_count} drops never recovered by present day")
            else:
                print(f"BIDEN: {len(self.biden_recovery)} major drops (>4%), none recovered by present day")
        else:
            print("BIDEN: No large drops (>4%) requiring recovery analysis")
            
        print("\n--- VOLATILITY ---")
        print(f"TRUMP: Average 30-day volatility: {self.trump_volatility_avg:.2f}%")
        print(f"BIDEN: Average 30-day volatility: {self.biden_volatility_avg:.2f}%")
        print(f"Difference: {(self.biden_volatility_avg - self.trump_volatility_avg):.2f}%")
    
    def generate_visualizations(self):
        """Generate visualizations for this index"""
        # 1. Drops comparison chart
        plt.figure(figsize=(10, 6))
        self.summary_counts.iloc[:, :-1].T.plot(kind='bar', color=['red', 'blue'])
        plt.title(f'{self.index_name} Drop Comparison: Trump vs Biden', fontsize=14)
        plt.ylabel('Number of Days', fontsize=12)
        plt.xlabel('Drop Severity', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'charts_multi/{self.ticker}_drops_comparison.png')
        
        # 2. Normalized performance comparison
        plt.figure(figsize=(10, 6))
        trump_norm = self.trump_period['Close'] / self.trump_period['Close'].iloc[0] * 100
        biden_norm = self.biden_period['Close'] / self.biden_period['Close'].iloc[0] * 100
        
        # Create day indices for x-axis
        trump_days = np.arange(len(trump_norm))
        biden_days = np.arange(len(biden_norm))
        
        plt.plot(trump_days, trump_norm, 'r-', linewidth=2, label='Trump')
        plt.plot(biden_days, biden_norm, 'b-', linewidth=2, label='Biden')
        
        plt.xticks([0, 252, 504, 756, 1008], ['Year 0', 'Year 1', 'Year 2', 'Year 3', 'Year 4'])
        plt.grid(True, alpha=0.3)
        plt.title(f'{self.index_name} Normalized Performance\n(Starting Point = 100)', fontsize=14)
        plt.ylabel('Normalized Price', fontsize=12)
        plt.xlabel('Time in Office', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'charts_multi/{self.ticker}_normalized.png')
        
        # 3. Volatility comparison
        plt.figure(figsize=(10, 6))
        plt.plot(self.trump_period.index, self.trump_volatility, 'r-', linewidth=1, alpha=0.7, label='Trump')
        plt.plot(self.biden_period.index, self.biden_volatility, 'b-', linewidth=1, alpha=0.7, label='Biden')
        plt.axhline(y=self.trump_volatility_avg, color='darkred', linestyle='--', label=f'Trump Avg: {self.trump_volatility_avg:.2f}%')
        plt.axhline(y=self.biden_volatility_avg, color='darkblue', linestyle='--', label=f'Biden Avg: {self.biden_volatility_avg:.2f}%')
        plt.title(f'{self.index_name} 30-Day Volatility Comparison', fontsize=14)
        plt.ylabel('Volatility (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'charts_multi/{self.ticker}_volatility.png')

def generate_combined_visualizations(analyzers):
    """Generate comparative visualizations across all indices"""
    # 1. Combined normalized performance under Trump
    plt.figure(figsize=(12, 6))
    for index_name, analyzer in analyzers.items():
        norm_data = analyzer.trump_period['Close'] / analyzer.trump_period['Close'].iloc[0] * 100
        days = np.arange(len(norm_data))
        plt.plot(days, norm_data, linewidth=2, label=index_name)
    
    plt.xticks([0, 252, 504, 756, 1008], ['Year 0', 'Year 1', 'Year 2', 'Year 3', 'Year 4'])
    plt.grid(True, alpha=0.3)
    plt.title('Normalized Index Performance During Trump Presidency\n(Starting Point = 100)', fontsize=16)
    plt.ylabel('Normalized Price', fontsize=14)
    plt.xlabel('Time in Office', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('charts_multi/combined_trump_performance.png')
    
    # 2. Combined normalized performance under Biden
    plt.figure(figsize=(12, 6))
    for index_name, analyzer in analyzers.items():
        norm_data = analyzer.biden_period['Close'] / analyzer.biden_period['Close'].iloc[0] * 100
        days = np.arange(len(norm_data))
        plt.plot(days, norm_data, linewidth=2, label=index_name)
    
    plt.xticks([0, 252, 504, 756, 1008], ['Year 0', 'Year 1', 'Year 2', 'Year 3', 'Year 4'])
    plt.grid(True, alpha=0.3)
    plt.title('Normalized Index Performance During Biden Presidency\n(Starting Point = 100)', fontsize=16)
    plt.ylabel('Normalized Price', fontsize=14)
    plt.xlabel('Time in Office', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('charts_multi/combined_biden_performance.png')
    
    # 3. Comparative volatility analysis
    volatility_data = {
        'Index': [],
        'Trump Avg Volatility': [],
        'Biden Avg Volatility': [],
        'Difference': []
    }
    
    for index_name, analyzer in analyzers.items():
        volatility_data['Index'].append(index_name)
        volatility_data['Trump Avg Volatility'].append(analyzer.trump_volatility_avg)
        volatility_data['Biden Avg Volatility'].append(analyzer.biden_volatility_avg)
        volatility_data['Difference'].append(analyzer.biden_volatility_avg - analyzer.trump_volatility_avg)
    
    vol_df = pd.DataFrame(volatility_data)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(vol_df['Index']))
    width = 0.35
    
    plt.bar(x - width/2, vol_df['Trump Avg Volatility'], width, label='Trump', color='red', alpha=0.7)
    plt.bar(x + width/2, vol_df['Biden Avg Volatility'], width, label='Biden', color='blue', alpha=0.7)
    
    plt.xticks(x, vol_df['Index'])
    plt.title('Average 30-Day Volatility Comparison', fontsize=16)
    plt.ylabel('Volatility (%)', fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts_multi/combined_volatility_comparison.png')
    
    # 4. Create consolidated drop frequency table
    print("\n" + "="*80)
    print("                      CONSOLIDATED DROP FREQUENCY ANALYSIS")
    print("="*80)
    
    # Format the dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    
    # Create a summary table of all drops across indices
    all_drops_data = {
        'Index': [],
        'Admin': [],
        'Tier 1 (1%-2%)': [],
        'Tier 2 (2%-4%)': [],
        'Tier 3 (>4%)': [],
        'Avg Recovery (days)': [],
        'Volatility (%)': []
    }
    
    for index_name, analyzer in analyzers.items():
        # Trump data
        all_drops_data['Index'].append(index_name)
        all_drops_data['Admin'].append('Trump')
        all_drops_data['Tier 1 (1%-2%)'].append(analyzer.trump_drops['Tier 1 (1%-2%)'])
        all_drops_data['Tier 2 (2%-4%)'].append(analyzer.trump_drops['Tier 2 (2%-4%)'])
        all_drops_data['Tier 3 (>4%)'].append(analyzer.trump_drops['Tier 3 (>4%)'])
        
        if not analyzer.trump_recovery.empty:
            recovery_days = analyzer.trump_recovery['Days to Recover'].dropna()
            if len(recovery_days) > 0:
                all_drops_data['Avg Recovery (days)'].append(f"{recovery_days.mean():.1f}")
            else:
                all_drops_data['Avg Recovery (days)'].append("N/A")
        else:
            all_drops_data['Avg Recovery (days)'].append("N/A")
            
        all_drops_data['Volatility (%)'].append(f"{analyzer.trump_volatility_avg:.2f}")
        
        # Biden data
        all_drops_data['Index'].append(index_name)
        all_drops_data['Admin'].append('Biden')
        all_drops_data['Tier 1 (1%-2%)'].append(analyzer.biden_drops['Tier 1 (1%-2%)'])
        all_drops_data['Tier 2 (2%-4%)'].append(analyzer.biden_drops['Tier 2 (2%-4%)'])
        all_drops_data['Tier 3 (>4%)'].append(analyzer.biden_drops['Tier 3 (>4%)'])
        
        if not analyzer.biden_recovery.empty:
            recovery_days = analyzer.biden_recovery['Days to Recover'].dropna()
            if len(recovery_days) > 0:
                all_drops_data['Avg Recovery (days)'].append(f"{recovery_days.mean():.1f}")
            else:
                all_drops_data['Avg Recovery (days)'].append("N/A")
        else:
            all_drops_data['Avg Recovery (days)'].append("N/A")
            
        all_drops_data['Volatility (%)'].append(f"{analyzer.biden_volatility_avg:.2f}")
    
    # Create and print the summary table
    summary_df = pd.DataFrame(all_drops_data)
    print("\n" + summary_df.to_string(index=False))
    
    # 5. Print comparative volatility table
    print("\n" + "="*80)
    print("                          VOLATILITY COMPARISON")
    print("="*80)
    print("\n" + vol_df.to_string(index=False))
    print("\n" + "="*80)

# Main execution
def main():
    analyzers = {}
    
    # Initialize and analyze each index
    for index_name, ticker in INDICES.items():
        analyzer = MarketAnalyzer(index_name, ticker)
        analyzer.download_data()
        analyzer.filter_periods()
        analyzer.analyze_drops()
        analyzer.print_summary()
        analyzer.generate_visualizations()
        analyzers[index_name] = analyzer
    
    # Generate combined visualizations
    print("\nGenerating combined visualizations...")
    generate_combined_visualizations(analyzers)
    
    print("\nAnalysis complete! Visualization files saved to the 'charts_multi' directory.")

if __name__ == "__main__":
    main() 