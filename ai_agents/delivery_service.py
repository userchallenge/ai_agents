# delivery_service.py
from flask import Flask, request, jsonify
import configparser

# config
CONFIG = "config/config.ini"
config = configparser.ConfigParser()
config.read(CONFIG)
API_KEY_OPENAI = config["API_KEYS"]["OPENAI"]

app = Flask(__name__)


@app.route("/get_delivery_estimate", methods=["POST"])
def get_delivery_estimate():
    data = request.json
    order_id = data.get("order_id")

    # Dummydata
    return jsonify(
        {
            "order_id": order_id,
            "estimated_delivery_date": "2025-08-20",
            "status": "OK",
            "warning": "Delay due to customs",
        }
    )


if __name__ == "__main__":
    app.run(port=5001)
