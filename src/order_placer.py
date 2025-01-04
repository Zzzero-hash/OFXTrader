import requests

OANDA_API_URL = "https://api-fxpractice.oanda.com/v3/accounts"

def place_order(config, instrument, units, stop_loss=None, take_profit=None):
    headers = {"Authorization": f"Bearer {config['access_token']}"}
    data = {
        "order": {
            "instrument": instrument,
            "units": str(units),
            "type": "MARKET",
            "timeInForce": "FOK"
        }
    }
    if stop_loss:
        data["order"]["stopLossOnFill"] = {"price": stop_loss}
    if take_profit:
        data["order"]["takeProfitOnFill"] = {"price": take_profit}

    response = requests.post(
        f"{OANDA_API_URL}/{config['account_id']}/orders",
        headers=headers,
        json=data
    )
    if response.status_code == 201:
        print("Order placed successfully.")
        return response.json()
    else:
        print(f"Failed to place order: {response.status_code} - {response.json().get('errorMessage', 'Unknown error')}")
        return None
