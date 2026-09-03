from __future__ import annotations

from ml.production.train import train_production_model
from ml.reserve.train import main as train_reserve


def main() -> None:
    train_reserve()
    print(train_production_model())


if __name__ == "__main__":
    main()
