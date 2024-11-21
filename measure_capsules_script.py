from tkinter.filedialog import askopenfilenames
import pickle
from datetime import datetime
import subprocess
import argparse


def main() -> list:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--shutdown_after",
        action="store_true",
        help="Shuts down the machine after the analysis is complete.",
    )
    args = parser.parse_args()

    datafiles = askopenfilenames()
    start_time = datetime.now()
    processes = []

    try:
        for file in datafiles:
            processes.append(
                datetime.now(),
                subprocess.run(
                    [
                        "py",
                        "measure_capsules.py",
                        "--reader",
                        "nd2reader",
                        "--datapath",
                        file,
                    ],
                    capture_output=True,
                    text=True,
                ),
                datetime.now(),
            )

    except KeyboardInterrupt:
        print("Processing interupted.")
        print("Saving processes states.")
        args.shutdown_after = False

    finally:
        with open(
            f"processes_{start_time.strftime("%d/%m/%y-%H_%M_%S_%f")}.pickle",
            "wb",
        ) as f:
            pickle.dump(processes, f)

        if args.shutdown_after:
            subprocess.run(["shutdown", "-s"])

        return processes


if __name__ == "__main__":
    processes = main()
