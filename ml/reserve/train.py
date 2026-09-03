from __future__ import annotations

from ml.reserve.train_prospectivity import train_grade, train_prospectivity, train_thickness


def main() -> None:
    print("================================")
    print("MANGAI RESERVE AI")
    print("================================")
    print(train_prospectivity())
    print(train_grade())
    print(train_thickness())
    print("RESERVE AI TRAINING COMPLETE")


if __name__ == "__main__":
    main()
