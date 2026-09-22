# semiautonomous_electron_microscopy

Automated SEM imaging and EDS spectroscopy on a **Thermo Fisher Scientific Phenom** desktop SEM,
driven from Python through the Phenom Programming Interface (PyPhenom).

The goal is to remove the operator from the repetitive part of an EDS survey: locate features in a
NavCam overview, drive the stage to each one, acquire SEM images, and collect EDS spot spectra at
chosen points — saving images, spectra and metadata into a timestamped project folder as it goes.

> **This code moves real hardware.** It drives the stage, sets high tension and beam spot size, and
> parks the beam in spot mode. Read a notebook before running it, and check the stage limits and
> beam settings against your sample.

---

## Requirements

### Hardware

- A Thermo Fisher Phenom desktop SEM reachable on the network.
- An EDS detector, for the spectroscopy steps.

### PyPhenom (the one dependency you cannot `pip install`)

`PyPhenom` — imported everywhere as `ppi` — is the **Phenom Programming Interface**, the proprietary
control SDK for the Phenom SEM. Nothing in this repo runs without it.

- It is **not on PyPI**. `pip install PyPhenom` will not give you a working package.
- Obtain the PPI SDK from **Thermo Fisher Scientific** or your Phenom support contact. It requires a
  **PPI licence tied to your instrument**.
- It installs as a compiled extension rather than a normal site-packages distribution. On this
  machine it lives at `C:\Python38\DLLs\PyPhenom.pyd` (PyPhenom **1.7.0**).
- Because it is a compiled `.pyd`, it is built against **one specific CPython version**. That is why
  this project is pinned to **Python 3.8** — the notebooks were developed and run on 3.8.0.

Check it before anything else:

```bash
py -3.8 -c "import PyPhenom; print(PyPhenom.__version__)"
```

### Python packages

```bash
py -3.8 -m pip install -r requirements.txt
```

One dependency is easy to miss: **`exspy`**. HyperSpy 2.x moved the EDS signal classes out of core,
so `set_signal_type("EDS_SEM")` in `AutoPhenom.writeSpectrum` only resolves when `exspy` is
installed alongside `hyperspy`. Installing `hyperspy` on its own will fail at spectrum-save time,
not at import time.

### Instrument credentials — `license.py`

Every notebook does `import license` and reads two module-level values from it:

| Name | Value |
|---|---|
| `license.PhenomUsername` | your Phenom instrument ID (the `MVE…` string) |
| `license.PhenomPassword` | the matching PPI password |

This file is **not in the repository** and you must create it yourself as `license.py`, anywhere on
`sys.path` — the project root is fine. Both values come from your PPI licence; if you do not have
them, ask whoever administers the instrument.

Because the file holds live instrument credentials, add `license.py` to `.gitignore` before your
first commit so it is never pushed. It is not currently listed there.

---

## Repository layout

| Path | What it is |
|---|---|
| `AutoPhenom.py` | Helper library wrapping PyPhenom: EDS spot acquisition, spectrum saving, SEM image capture, plot annotation, project folders. |
| `segment_circles.py` | Hough-transform detection of circular features in a NavCam image; converts pixel centres to stage coordinates and writes `circles_metadata.json`. |
| `copper_case_study.ipynb` | End-to-end run: NavCam overview → circle detection → per-circle SEM imaging → EDS at points around each circle. |
| `NickelSphereImageSegmentation.ipynb` | Contour-based segmentation of nickel spheres, then EDS spot measurements along each particle radius. |
| `requirements.txt` | Pinned package set verified on the instrument PC. |

The notebooks are the programs; the two `.py` files exist only to serve them.

### `AutoPhenom.py`

| Function | Purpose |
|---|---|
| `create_timestamped_folder(base_path, prefix="Run")` | Make a `Run_YYYY-MM-DD_HH-MM-SS` project folder. |
| `AcquireSEMImage_at_current_location(path, filename, size_px, phenom)` | Capture a 16-frame SEM image at the current stage position; saves raw and databar TIFFs, returns the raw path. |
| `getSpotSpectrum(i, j, size, SavePath, MainPath, dpp, address, phenom, sample_name, dwell_time)` | Park the beam at normalised image coordinate `(i, j)`, acquire an EDS spectrum for `dwell_time` seconds, save it, return it as a NumPy array. |
| `setSpot_test(phenom, imageSize, position)` | Put the column into spot mode at a given position. Called by `getSpotSpectrum`. |
| `writeSpectrum(spectrum, filename, address, sample_name, dwell_time)` | Save a spectrum as `.msa`, `.npy` and `.hdf5`, with sample/instrument/dwell metadata attached to the HDF5 dataset. Called by `getSpotSpectrum`. |
| `initialize_marker_plot(image_path)` | Open an SEM image as a matplotlib figure ready for annotation. |
| `add_marker(ax, img_shape, position)` | Mark and label a normalised coordinate on that figure. |
| `save_annotated_image(fig, image_path)` | Write the annotated figure out as `annotated_<name>`. |

### `segment_circles.py`

`segment_circles(image_path, d_min_mm, d_max_mm, hfw_metres, ...)` detects circular features within
a given diameter range, filters them by circularity, edge clipping and non-maximum suppression, and
returns `(annotated_image, mask, circles, metadata_path)`. Given an `output_dir` it writes an
annotated image plus `circles_metadata.json` containing each circle's pixel centre, radius, diameter
in mm and **stage coordinates in metres** — which is what the imaging notebooks drive from.

Coordinate conversion needs the true horizontal field width, so pass `phenom.GetHFW()` rather than
the value you requested.

---

## Running a session

1. Create `license.py` with your instrument credentials.
2. Confirm `import PyPhenom` works under Python 3.8.
3. Set the project output directory near the top of the notebook — currently a hard-coded absolute
   path such as `r"C:\Ankush_Phenom_Codes\Data\microscopy_automation\test_copper"`. **Change this to
   a path that exists on your machine.**
4. Adjust the acquisition parameters for your sample: NavCam HFW, the `d_min_mm` / `d_max_mm`
   feature-size window, `hough_param2` and `min_circularity` for detection, then high tension, spot
   size, SEM HFW and EDS dwell time.
5. Run the cells in order. Each run writes into its own timestamped folder.

### Output

```
Run_2026-09-22_14-38-00/
├── navcam.tiff                     # NavCam overview
├── navcam_databar.tiff
├── navcam_annotated.tiff           # detected circles, numbered
├── circles_metadata.json           # centres, diameters, stage coordinates
├── sem_metadata.json               # one record per circle/location visited
└── sem_images/
    └── circle_01/
        └── <location_label>/       # e.g. an edge point around the circle
            └── hfw_30um/
                ├── circle_01_hfw_30um.tiff
                └── circle_01_hfw_30um_databar.tiff
```

EDS spectra are written per measurement point as `.msa`, `.npy` and `.hdf5`; the HDF5 file carries
sample name, instrument address and dwell time as dataset attributes.

---

## Known rough edges

- **Hard-coded paths.** Output directories and some stage offsets are literals in the notebooks, not
  parameters. Change them before running.
- **`os.chdir` side effects.** `getSpotSpectrum` and `AcquireSEMImage_at_current_location` change the
  process working directory and rely on the caller passing a `MainPath` to return to. Use absolute
  paths for anything you open yourself.
- **No tests.** The code can only be exercised against a real instrument, so there is no test suite
  and no way to validate a change offline beyond checking that it imports.
- **Stage settling** is handled with fixed `time.sleep` calls rather than a position check.

## Licence

Released under **CC0 1.0 Universal** (public domain dedication) — see `LICENSE`.
