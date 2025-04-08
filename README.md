# Market Performance Analysis: Trump vs Biden Administration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python-based analysis comparing stock market performance during the Trump and Biden presidential administrations. The analysis focuses on market drops, recovery patterns, and volatility across multiple indices.

![Market Analysis Preview](https://placeholder-for-screenshot.png)

## 📊 Features

1. **Multi-Index Historical Analysis**: Downloads and analyzes data from:
   - S&P 500
   - Nasdaq Composite
   - Dow Jones Industrial Average

2. **Presidential Period Comparison**:
   - Trump: January 20, 2017 to January 19, 2021
   - Biden: January 20, 2021 to present

3. **Market Drop Classification**:
   - Tier 1 (Small): 1% to 2% drops
   - Tier 2 (Medium): 2% to 4% drops
   - Tier 3 (Large): >4% drops

4. **Advanced Metrics**:
   - Recovery Time Analysis: Calculates how long markets recover from significant drops
   - Volatility Measurement: Quantifies and compares market volatility
   - Annualized comparisons to account for different time periods

5. **Rich Visualizations**: Generates multiple charts for easy interpretation of results

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/market-performance-analysis.git
   cd market-performance-analysis
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📝 Usage

The repository includes three scripts with increasing levels of complexity:

1. **Basic Analysis** - S&P 500 only:
   ```bash
   python trumpvsbiden.py
   ```

2. **Enhanced Analysis** - S&P 500 with additional metrics:
   ```bash
   python market_analysis.py
   ```

3. **Comprehensive Multi-Index Analysis** - All three major indices:
   ```bash
   python multi_index_analysis.py
   ```

All visualizations are saved to either the `charts/` or `charts_multi/` directories.

## 📈 Sample Output

The analysis generates tables and visualizations including:

- Drop frequency comparison between administrations
- Normalized performance charts (indexed to 100 at start)
- Recovery time statistics for significant market drops
- Volatility comparison charts
- Cross-index performance metrics

## 📋 Scripts Overview

### trumpvsbiden.py
Basic version focused on the S&P 500 index with simple visualizations.

### market_analysis.py
Enhanced analysis of the S&P 500 with additional metrics:
- More detailed recovery analysis
- Volatility measurements
- Annualized metrics
- Additional visualizations

### multi_index_analysis.py
Comprehensive analysis of three major indices:
- Object-oriented implementation for better code structure
- Cross-index comparisons
- Consolidated summary tables
- Combined visualizations for easier comparison

## 🛠️ Customization

You can customize the analysis by modifying:

- Time periods analyzed
- Drop tier thresholds
- Indices analyzed (by adding to the `INDICES` dictionary)
- Visualization styles and formats
- Recovery threshold parameters

## 📚 Methodology

1. **Data Collection**: Historical price data is retrieved using the yfinance API
2. **Period Filtering**: Data is filtered to match presidential terms
3. **Drop Analysis**: Daily price movements are categorized into severity tiers
4. **Recovery Calculation**: For each major drop, the time to recover to pre-drop levels is calculated
5. **Volatility Measurement**: 30-day rolling standard deviation of returns

## ⚠️ Disclaimer

This analysis is for educational and informational purposes only and should not be considered financial advice. Market performance is influenced by numerous factors beyond presidential administration policies, including:

- Federal Reserve policy
- Global economic conditions
- Industry-specific developments
- Technological changes
- Natural disasters and pandemics
- Geopolitical events

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 