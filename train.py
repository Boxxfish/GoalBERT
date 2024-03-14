"""
Performs the training for GoalBERT.
"""

from goalbert.config import GoalBERTConfig


def main():
    config = GoalBERTConfig.parse_args()
    print(config.flat_dict())


if __name__ == "__main__":
    main()
