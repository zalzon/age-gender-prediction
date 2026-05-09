from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.gui.main_window import AgeGenderPredictionApp


def main() -> None:
    app = AgeGenderPredictionApp()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApp closed.")
        sys.exit(0)
