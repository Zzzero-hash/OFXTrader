# OFXTrader

A tool for automated trading and financial data management using the OANDA API.

## Overview

This trading bot is designed to handle automated trading, financial data fetching, and account management using the OANDA API.

## Features

- Automated trading with reinforcement learning
- Account details fetching and live display
- Historical data fetching
- Trade tracking and reporting
- Risk management with stop loss and profit targets

## Goals

- Streamline automated forex trading using reinforcement learning.
- Provide easy retrieval of historical and real-time market data.
- Integrate advanced technical analysis indicators and ML models.
- Ensure the bot runs until it hits a profit target or a stop loss.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OFXTrader.git

# Navigate to the project directory
cd OFXTrader

# Install the required packages
pip install -r requirements.txt
```

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up your OANDA credentials in `config/config.json`.

3. Run the main bot:

   ```bash
   python auto_trader.py --granularity H1 --instrument EUR_USD --max-profit-percent 0.1 --max-loss-percent 0.1
   ```

## Usage

1. **Configuration**: Ensure you have your OANDA account ID and API key. The configuration will be saved in the `config/config.json` file.

2. **Run the Auto Trader**: Start the trading bot by running the following command:

    ```bash
    python auto_trader.py --granularity H1 --instrument EUR_USD --max-profit-percent 0.1 --max-loss-percent 0.1
    ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.