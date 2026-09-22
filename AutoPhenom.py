# Autonomous Control of the Phenom SEM

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import hyperspy.api as hs
import PyPhenom as ppi
from datetime import datetime


# Acquire a spot spectrum
def getSpotSpectrum(
        i,j,
        size,
        SavePath,
        MainPath,
        dpp,
        address,
        phenom,
        sample_name,
        dwell_time):

    # Change coordinate system
    i_new, j_new = i - 0.5, j - 0.5
    setSpot_test(phenom, size, (i_new, j_new))

    # Perform EDS measurement
    dpp.ClearSpectrum()
    dpp.Start()
    time.sleep(dwell_time)
    dpp.Stop()

    # Save spectrum data
    os.chdir(SavePath)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    spectrum = writeSpectrum(dpp.GetSpectrum(), f'Spectrum at ({i:.2f}, {j:.2f}) taken at {timestamp}', address = address, sample_name = sample_name, dwell_time = dwell_time)
    os.chdir(MainPath)

    return spectrum # returns a spectrum in numpy array format

# Set spot for spot spectrum measurement
def setSpot_test(phenom, imageSize, position):
    acqScanParams = ppi.ScanParams()
    acqScanParams.size = ppi.Size(imageSize[0],imageSize[1])
    acqScanParams.detector = ppi.DetectorMode.All
    acqScanParams.nFrames = 16
    acqScanParams.hdr = False
    acqScanParams.center = ppi.Position(position[0], position[1]) # This works for square images only. See manual to implement rectangular images

    acqScanParams.scale = 0.001 # Set this to be small relative to image horizontal field width
    mode = ppi.SemViewingMode(ppi.ScanMode.Spot, acqScanParams)

    phenom.SetSemViewingMode(mode)

# Write spectrum to file form
def writeSpectrum(spectrum, filename, address, sample_name, dwell_time):
    dict0 = {'offset': spectrum.offset, 'scale': spectrum.dispersion, 'size': len(spectrum.data), 'units': 'eV'}
    s = hs.signals.Signal1D(np.array(spectrum.data), axes=[dict0])
    s.set_signal_type("EDS_SEM")

    s.save(f'{filename}.msa', encoding = 'utf8')

    spectrum = np.array(spectrum.data)
    print(f'From inside writespectrum, we create a spectrum of shape {np.shape(spectrum)}')
    np.save(filename, spectrum)
    with h5py.File(f"{filename}.hdf5", "w") as f:
        # Create the dataset
        dset = f.create_dataset(f"{filename}_dataset", data=spectrum.data)

        # Add metadata (attributes) to the dataset
        dset.attrs["Sample Name"] = f'{sample_name}'
        dset.attrs['Instrument Address'] = f'{address}'
        dset.attrs['Dwell/Acquisition Time'] = f'{dwell_time}'
        dset.attrs['Date and Time of Acquisition'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return spectrum

# Initialize marker plot
def initialize_marker_plot(image_path):
    """
    Initializes the plot with the SEM image and returns the figure and axis objects.
    """
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis('off')  # Turn off axis labels
    return fig, ax, img.shape  # Return image shape for coordinate scaling

# Add marker
def add_marker(ax, img_shape, position):
    """
    Adds a marker to an existing axis using normalized coordinates.
    """
    height, width = img_shape[0], img_shape[1]
    x_coord, y_coord = position
    x_px = x_coord * width
    y_px = y_coord * height
    ax.plot(x_px, y_px, marker='.', color='red', markersize=5, markeredgewidth=2)

    # Add text label with normalized coordinates
    label = f'({x_coord:.2f}, {y_coord:.2f})'
    ax.text(x_px + 0, y_px - 5, label, color='red', fontsize=4, weight='bold')  # Adjust offset and styling

# save annotated image
def save_annotated_image(fig, image_path):
    """
    Saves the annotated image with all markers added.
    """
    out_path = f'annotated_{os.path.basename(image_path)}'
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure when done
    return out_path

# Create a folder with the current datetime
def create_timestamped_folder(base_path, prefix="Run"):
    # Get the current time as a string
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the folder name (e.g., "Run_2025-08-07_14-38-00")
    folder_name = f"{prefix}_{timestamp}"

    # Full path
    full_path = os.path.join(base_path, folder_name)

    # Create the folder
    os.makedirs(full_path, exist_ok=True)

    return full_path


def AcquireSEMImage_at_current_location(path, filename, image_side_length_in_pixels, phenom):
    phenom.MoveToSem()
    # phenom.SemAutoFocus() # Autofocuses the SEM (which is the same as finding the optimal working distance)
    # phenom.SemAutoContrastBrightness()

    # Change to directory where the image will be saved
    os.chdir(path)
    # Acquire SEM image
    acqScanParams = ppi.ScanParams()
    acqScanParams.size = ppi.Size(image_side_length_in_pixels, image_side_length_in_pixels) # Change to 256 x 256 for SLADS
    acqScanParams.detector = ppi.DetectorMode.All
    acqScanParams.nFrames = 16 # number of frames to average for signal to noise improvement.
    acqScanParams.hdr= False # a Boolean to use High Dynamic Range mode
    acqScanParams.scale = 1.0
    acq = phenom.SemAcquireImage(acqScanParams)
    acq.metadata.displayWidth = 0.5
    acq.metadata.dataBarLabel = "Label"
    acqWithDatabar = ppi.AddDatabar(acq)

     # Save both versions
    raw_path = os.path.join(path, f"{filename}.tiff")
    databar_path = os.path.join(path, f"{filename}withDatabar.tiff")

    ppi.Save(acq, raw_path)
    ppi.Save(acqWithDatabar, databar_path)

    # Return full file path of the raw image (or databar image if you prefer)
    return raw_path
