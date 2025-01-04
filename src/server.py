from flask import Flask, request, jsonify
from .config_utils import load_config
from .order_placer import place_order

app = Flask(__name__)

# Load config at the module level or within an initializer
config = load_config()

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    action = data.get("action")

    if action == "buy":
        place_order(config, "EUR_USD", units=100)
    elif action == "sell":
        place_order(config, "EUR_USD", units=-100)

    return jsonify({"status": "success"})

def run_server():
    # Optionally, allow configuring host/port
    app.run(host="0.0.0.0", port=5000, debug=True)
