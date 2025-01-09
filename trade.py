from oandapyV20 import API
from oandapyV20.endpoints import orders
import json
import logging
from train import load_model  # Retain only model loading utility

class TradingBot:
    def __init__(self, model_name, instrument, granularity, start_date, end_date, access_token, max_profit_percent, max_loss_percent):
        self.model_name = model_name
        self.instrument = instrument
        self.granularity = granularity
        self.start_date = start_date
        self.end_date = end_date
        self.access_token = access_token
        self.max_profit_percent = max_profit_percent
        self.max_loss_percent = max_loss_percent
        self.is_live = False
        # Initialize trade metrics
        self.trade_metrics = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0,
            'average_profit': 0.0
        }
        self.api = API(access_token=access_token, environment="live")  # Initialize OANDA API

        # Load the pre-trained model
        self.model = load_model(self.model_name)

    def start_live_trading(self):
        self.is_live = True
        logging.info(f"Starting live trading with model: {self.model_name}")
        print(f"Starting live trading with model: {self.model_name}")
        # Implement live trading loop or initiate trading sessions here
        # Example: Connect to WebSocket for real-time data and execute trades
        # ...existing live trading code...

    def stop_live_trading(self):
        self.is_live = False
        logging.info("Stopping live trading")
        print("Stopping live trading")
        # Output final trade metrics
        print("Final Trade Metrics:")
        print(f"Total Trades: {self.trade_metrics['total_trades']}")
        print(f"Wins: {self.trade_metrics['wins']}")
        print(f"Losses: {self.trade_metrics['losses']}")
        print(f"Total Profit: {self.trade_metrics['total_profit']:.2f}")
        if self.trade_metrics['total_trades'] > 0:
            self.trade_metrics['average_profit'] = self.trade_metrics['total_profit'] / self.trade_metrics['total_trades']
            print(f"Average Profit per Trade: {self.trade_metrics['average_profit']:.2f}")

    def execute_trade(self, trade_signal):
        if not self.is_live:
            print("Trading bot is not live. Cannot execute trade.")
            return
        # Execute trade using OANDA API
        order = {
            "order": {
                "instrument": trade_signal.get('symbol', self.instrument),
                "units": trade_signal.get('quantity', 0),
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        try:
            r = orders.OrderCreate(self.api, trade_signal.get('account_id', self.instrument), data=order)
            response = self.api.request(r)
            logging.info(f"Trade executed: {response}")
            print(f"Trade executed: {response}")
            
            # Update trade metrics based on response
            # Assuming response contains profit information
            profit = float(response.get('orderFillTransaction', {}).get('price', 0.0))  # Replace with actual profit extraction
            outcome = 'win' if profit > 0 else 'loss'
            
            # Update trade metrics
            self.trade_metrics['total_trades'] += 1
            if outcome == 'win':
                self.trade_metrics['wins'] += 1
                self.trade_metrics['total_profit'] += profit
            elif outcome == 'loss':
                self.trade_metrics['losses'] += 1
                self.trade_metrics['total_profit'] += profit
        except Exception as e:
            logging.error(f"Failed to execute trade: {e}")
            print(f"Failed to execute trade: {e}")

if __name__ == "__main__":
    selected_model = "YourModelName"  # Replace with the user's selected model
    instrument = "EUR_USD"
    granularity = "H1"
    start_date = "2024-01-01T00:00:00Z"
    end_date = "2024-12-31T00:00:00Z"
    access_token = "your_access_token"
    max_profit_percent = 0.1
    max_loss_percent = 0.1

    bot = TradingBot(selected_model, instrument, granularity, start_date, end_date, access_token, max_profit_percent, max_loss_percent)
    bot.start_live_trading()
    # Example trade signal
    trade_signal = {"action": "buy", "quantity": 10, "symbol": "AAPL", "account_id": "your_account_id"}
    bot.execute_trade(trade_signal)
    bot.stop_live_trading()