import random

# Path to the file with filtered paths
input_file_path = 'data/zod_dataset/data_lists/randomly_filtered_output_paths.txt'

# Read all paths from the file
with open(input_file_path, 'r') as file:
    paths = file.readlines()

# Shuffle the list to randomize the order
random.shuffle(paths)

# Calculate the split index for 20% validation
split_index = int(0.2 * len(paths))

# Split the paths into validation and training
validation_paths = paths[:split_index]
training_paths = paths[split_index:]

# Write the validation paths to a text file
with open('validation.txt', 'w') as file:
    for path in validation_paths:
        file.write(path)

# Write the training paths to a text file
with open('training.txt', 'w') as file:
    for path in training_paths:
        file.write(path)

print("Data split into validation (20%) and training (80%) sets.")
