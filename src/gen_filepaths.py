import os
import random

def get_files_with_extension(folder_path, extension):
    """ Gets the path of all files with a given extension in a folder recursively """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        # Only process directories that contain 'image_02'
        if 'image_02' in root:
            for file in files:
                if file.endswith(extension):
                    file_paths.append(os.path.join(root, file))
    return file_paths

def split_data(file_paths, test_ratio=0.10, train_ratio=0.85, val_ratio=0.05):
    """ Splits file paths into three groups based on given ratios """
    # Shuffle the list randomly
    random.shuffle(file_paths)
    
    # Calculate the number of files for each group
    total_files = len(file_paths)
    test_count = int(total_files * test_ratio)
    train_count = int(total_files * train_ratio)
    val_count = total_files - test_count - train_count  # Remaining files for validation

    # Split the file paths
    test_files = file_paths[:test_count]
    train_files = file_paths[test_count:test_count + train_count]
    val_files = file_paths[test_count + train_count:]
    
    return train_files, test_files, val_files

def write_to_file(file_paths, output_file):
    """ Writes the file paths to a text file """
    with open(output_file, 'w') as f:
        for line in file_paths:
            f.write(f"{line}\n")

# Define folder path and extension
folder_path = "/home/skushwaha/DLG2_p3/src/Dataset/"
extension = ".png"

# Get all .png files in the folder and subfolders that include 'image_02' in their directory path
paths = get_files_with_extension(folder_path, extension)

# Split the data into 85% train, 10% test, and 5% validation
train_files, test_files, val_files = split_data(paths)

# Write the file lists to their respective text files
write_to_file(train_files, '/home/skushwaha/DLG2_p3/src/Src/list/eigen_train_list.txt')
write_to_file(test_files, '/home/skushwaha/DLG2_p3/src/Src/list/eigen_test_list.txt')
write_to_file(val_files, '/home/skushwaha/DLG2_p3/src/Src/list/eigen_val_list.txt')

print("Files have been successfully split and saved!")