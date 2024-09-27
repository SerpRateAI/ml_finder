import numpy as np
import sys

# Function to modify rows in a 2D array
def remove_resonance(list_of_arrays):
    modified_arrays = []
    for array in list_of_arrays:
        modified_array = array.copy()  # Create a copy to avoid modifying the original array
        for i, row in enumerate(array):
            if np.sum(row) > (len(row) // 2):  # Check if more than half of the row is True
                modified_array[i] = False  # Set the entire row to False if condition is met
        modified_arrays.append(modified_array)
    return modified_arrays

def red_resonance(list_of_arrays):
    modified_arrays = []
    for array in list_of_arrays:
        modified_array = array.copy()  # Create a copy to avoid modifying the original array
        for i, row in enumerate(array):
            if np.sum(row) > (len(row) // 2):  # Check if more than half of the row is True
                modified_array[i] = 2  # Set the entire row to False if condition is met
        modified_arrays.append(modified_array)
    return modified_arrays


window = int(sys.argv[1])
threshold = int(sys.argv[2])
bin_specs_arr = np.load(f'/fp/projects01/ec332/data/altered_spectrograms/bin_spec_{window}_{threshold}.npz')['spectrograms']

no_res = remove_resonance(bin_specs_arr)
np.savez(f"/fp/projects01/ec332/data/altered_spectrograms/bin_spec_no_res_{window}_{threshold}.npz", spectrograms = no_res)

int_array = bin_specs_arr.astype(np.int32)
int_array_red_res = red_resonance(list(int_array))
int_array_red_res = np.array(int_array_red_res)
np.savez(f"/fp/projects01/ec332/data/altered_spectrograms/bin_spec_red_res_{window}_{threshold}.npz", spectrograms=int_array_red_res)