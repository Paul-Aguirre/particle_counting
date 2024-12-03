from tkinter.filedialog import askopenfilenames
from tkinter.messagebox import askyesno
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
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Runs the measure_capsules script with the '--recompute' flag.",
    )
    args = parser.parse_args()
    datafiles = []
    choose_again = True

    while choose_again:
        datafiles += [file for file in askopenfilenames()]
        choose_again = askyesno(
            title="Continue?",
            message="Do you wish to select other files?",
        )

    start_time = datetime.now()
    processes = []

    command = [
        "py",
        "measure_capsules.py",
        "--recompute",
        "--reader",
        "nd2reader",
        "--datapath",
    ]
    if not args.recompute:
        command.remove("--recompute")

    for file in datafiles:
        try:
            processes.append(
                (
                    datetime.now(),
                    subprocess.run(
                        command + [file],
                        capture_output=True,
                        text=True,
                    ),
                    datetime.now(),
                )
            )

        except KeyboardInterrupt:
            print("Processing interupted.")
            args.shutdown_after = False
            break

        finally:
            print(f"Saving process state for file '{file}'.")
            with open(
                f"processes_{start_time.strftime("%d_%m_%y-%H_%M_%S_%f")}.pickle",
                "wb",
            ) as f:
                pickle.dump((datafiles, processes), f)

    if args.shutdown_after:
        subprocess.run(["shutdown", "-s"])

    return processes


if __name__ == "__main__":
    processes = main()
