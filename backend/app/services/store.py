_db = {}


def save_prediction(prediction_id: str, data: dict):
    _db[prediction_id] = data


def get_prediction(prediction_id: str):
    return _db.get(prediction_id)
