import os
import shutil

# Create class folders if they don't exist
for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    os.makedirs(f"data/test/{letter}", exist_ok=True)
os.makedirs("data/test/nothing", exist_ok=True)
os.makedirs("data/test/space", exist_ok=True)
os.makedirs("data/test/del", exist_ok=True)

# Move each test image into its corresponding class folder
for filename in os.listdir("data/test"):
    if filename.endswith("_test.jpg"):
        # Extract class name (e.g., 'A' from 'A_test.jpg')
        class_name = filename.split("_")[0]
        # Move the file
        shutil.move(f"data/test/{filename}", f"data/test/{class_name}/{filename}")

print("Test images organized into class folders.") 