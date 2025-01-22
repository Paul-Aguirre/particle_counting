from pathlib import Path
from tkinter.filedialog import askopenfilenames
from tkinter.messagebox import askyesno

from files_io import (
    get_metadata_nd2_altenartive,
    save_metadata,
    get_save_path,
)


def main() -> None:

    paths: tuple[str] = []
    choose_again: bool = True

    while choose_again:
        paths += [Path(file) for file in askopenfilenames()]
        choose_again = askyesno(
            title="Continue?",
            message="Do you wish to select other files?",
        )

    for path in paths:
        metadata_dict = get_metadata_nd2_altenartive(path)
        save_metadata(
            metadata_dict=metadata_dict,
            file=get_save_path(
                datapath=path,
                suffix="metadata",
                ext="pickle",
            ),
        )


if __name__ == "__main__":
    main()
