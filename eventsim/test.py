import h5py
# import hdf5plugin
import os

# Set the environment variable programmatically to point to the plugin directory.
# This must be done *before* h5py opens a file that needs the plugin.
# Note: This assumes you run the script from the 'eventbased' directory.
# plugin_path = os.path.abspath('hdf5_ecf/build/Release')
# os.environ['HDF5_PLUGIN_PATH'] = plugin_path

# On Windows with Python 3.8+, you may also need to add the path to the DLL search path.
# try:
#     os.add_dll_directory(plugin_path)
# except AttributeError:
#     # os.add_dll_directory is not available in older Python versions.
#     # The HDF5_PLUGIN_PATH should be sufficient.
#     pass

try:
    with h5py.File("driving_data.hdf5", "r") as f:
      print("Successfully opened recording.hdf5")
      evts = f["CD"]["events"]
      print(evts)
      # Run the query from your original script
      filtered_evts = evts[(evts['t']<49330) & (evts['t']>48270) & (evts['x']<300) & (evts['y']>300)]
      print(f"Found {len(filtered_evts)} events matching the criteria.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("\nPlease ensure that 'recording.hdf5' exists and the ECF plugin is working correctly.")