import os

# Folder paths
clean_folder = "dataset/train/clean"
dirty_folder = "dataset/train/dirty"

# Count images
clean_count = len(os.listdir(clean_folder))
dirty_count = len(os.listdir(dirty_folder))

# Print results
print("Clean images:", clean_count)
print("Dirty images:", dirty_count)

print("Total images:", clean_count + dirty_count)


# Folder paths
clean_folder2 = "dataset/test/clean"
dirty_folder2 = "dataset/test/dirty"

# Count images
clean_count2 = len(os.listdir(clean_folder2))
dirty_count2 = len(os.listdir(dirty_folder2))

# Print results
print("Clean images:", clean_count2)
print("Dirty images:", dirty_count2)

print("Total images:", clean_count2 + dirty_count2)