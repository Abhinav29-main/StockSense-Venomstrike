import json

class ArmorIQShield:
    def __init__(self, policy_path="policy.json"):
        with open(policy_path) as f:
            self.policy = json.load(f)

    def validate_intent(self, ticker, amount, confidence):
        if ticker.upper() in [t.upper() for t in self.policy["restricted_tickers"]]:
            return False, f"Blocked: {ticker} is restricted"
        if amount > self.policy["max_trade_amount_usd"]:
            return False, f"Blocked: ${amount} exceeds limit ${self.policy['max_trade_amount_usd']}"
        if confidence < self.policy["min_ml_confidence_percent"]:
            return False, f"Blocked: confidence {confidence}% < {self.policy['min_ml_confidence_percent']}%"
        return True, "Shield passed"