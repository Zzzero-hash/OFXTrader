import requests
import oandapyV20

OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"

def get_oanda_client(config):
    """
    Create an OANDA API client using the provided configuration.
    """
    return oandapyV20.API(access_token=config["access_token"])

def fetch_instrument_list(config):
    """
    Fetch the list of available instruments from the OANDA API.
    """
    headers = {"Authorization": f"Bearer {config['access_token']}"}

    try:
        print("Fetching instrument list...")
        response = requests.get(f"{OANDA_API_URL}/instruments", headers=headers)
        if response.status_code == 200:
            instruments = response.json()["instruments"]
            print(f"Fetched {len(instruments)} instruments.")
            return instruments
        else:
            print("Failed to fetch instrument list.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []