from flask import Flask, request, jsonify
import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
from src.key_handler import decrypt_data
import logging

app = Flask(__name__)

# Initialize OANDA API client
api_key, account_id = decrypt_data()
client = oandapyV20.API(access_token=api_key)

def get_account_balance():
    r = accounts.AccountDetails(account_id)
    response = client.request(r)
    balance = float(response['account']['balance'])
    return balance

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid data'}), 400

    try:
        # Extract trade details from the JSON payload
        instrument = data['instrument']
        action = data['action']  # 'open' or 'close'
        order_type = data.get('order_type')
        price = data.get('price')
        stop_loss = data.get('stop_loss')
        take_profit = data.get('take_profit')
        position_direction = data.get('position_direction')  # 'long' or 'short'

        if action == 'open':
            # Get account balance and calculate units for 50% of the balance
            balance = get_account_balance()
            units = int((balance * 0.5) / price)

            # Create the order request
            order_data = {
                "order": {
                    "instrument": instrument,
                    "units": units,
                    "type": order_type,
                    "price": price,
                    "stopLossOnFill": {
                        "price": stop_loss
                    } if stop_loss else None,
                    "takeProfitOnFill": {
                        "price": take_profit
                    } if take_profit else None,
                    "timeInForce": "GTC"
                }
            }

            # Remove None values from the order data
            order_data['order'] = {k: v for k, v in order_data['order'].items() if v is not None}

            # Send the order request to OANDA
            r = orders.OrderCreate(account_id, data=order_data)
            response = client.request(r)
            logging.info(f"Order response: {response}")

        elif action == 'close':
            # Close the position for the given instrument and direction
            if position_direction == 'long':
                r = positions.PositionClose(account_id, instrument=instrument, data={"longUnits": "ALL"})
            elif position_direction == 'short':
                r = positions.PositionClose(account_id, instrument=instrument, data={"shortUnits": "ALL"})
            else:
                return jsonify({'error': 'Invalid position direction'}), 400

            response = client.request(r)
            logging.info(f"Close position response: {response}")

        else:
            return jsonify({'error': 'Invalid action'}), 400

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error processing webhook: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
