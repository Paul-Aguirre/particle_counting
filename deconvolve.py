from pathlib import Path
from tkinter.filedialog import askopenfilename
import tomllib

from matplotlib import pyplot as plt
import numpy as np
from skimage import restoration

from files_io import load_image_stack, check_config
from multi_slice_viewer import MultiSliceViewer


def deconvolve(
    datapath: str | Path,
    psf_path: str | Path,
    num_iter: int = 1,
):
    psf = np.load(psf_path)

    datapath, _, configpath = check_config(datapath)

    with open(configpath, "rb") as f:
        config = tomllib.load(f)

    stack, _ = load_image_stack(
        path=str(datapath),
        zstart=config["stack_start"],
        zstop=config["stack_stop"],
    )

    deconv_stack = restoration.richardson_lucy(
        image=stack,
        psf=psf,
        num_iter=num_iter,
    )

    return deconv_stack


if __name__ == "__main__":
    psf_path = (
        r"C:\Users\aguirrep\Documents\microscopie\fluo_tests\comptage"
        r"\restored_psf.npy"
    )
    datapath = Path(
        r"C:\Users\aguirrep\Documents\microscopie"
        r"\fluo_tests\comptage\data\calib-caps\1.5pc-20-caps1.nd2"
    )
    # psf_path = askopenfilename(title="Select a PSF file")
    # datapath = Path(askopenfilename(title="Choose an image stack"))
    num_iter = 100
    deconv_stack = deconvolve(datapath, psf_path, num_iter)
    # plotting
    print(f"num_inter = {num_iter}")
    # viewer = MultiSliceViewer(lognorm=False)
    # viewer.plot(volume=deconv_stack)
    # viewer.show()
