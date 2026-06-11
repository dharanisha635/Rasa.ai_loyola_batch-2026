import os
import hashlib

def remove_duplicates(folder):
    hashes = {}
    
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        try:
            with open(path, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()

            if filehash in hashes:
                print("Duplicate removed:", filename)
                os.remove(path)
            else:
                hashes[filehash] = filename

        except Exception as e:
            print("Error:", filename, e)

# Clean folder
remove_duplicates("dataset/train/clean")

# Dirty folder
remove_duplicates("dataset/train/dirty")

print("Duplicate removal completed!")