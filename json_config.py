import json

from utils import HParams


def get_hparams_from_json(data):
    config = json.loads(data)
    hparams = HParams(**config)
    return hparams


def load_json_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data
