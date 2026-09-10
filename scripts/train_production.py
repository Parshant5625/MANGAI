from __future__ import annotations

import json

from ml.production.train import train_production_models

if __name__ == "__main__":
    print(json.dumps(train_production_models(), indent=2))
