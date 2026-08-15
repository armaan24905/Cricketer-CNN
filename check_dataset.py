import os
from PIL import Image

DATASET_PATH = r"C:\Users\radia\OneDrive\Desktop\🚀Armaan📊\Cricketer_CNN\results\train"
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
bad_files = []
total_files = 0
valid_files = 0
print("\nChecking dataset...\n")
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        file_path = os.path.join(root, file)
        extension = os.path.splitext(file)[1].lower()
        if extension not in valid_extensions:
            print("Invalid extension:", file_path)
            bad_files.append(file_path)
            continue
        total_files += 1
        try:
            with Image.open(file_path) as img:
                img.verify()
            valid_files += 1
        except Exception as e:
            print("\n❌ BAD IMAGE:")
            print(file_path)
            print("Error:", e)
            bad_files.append(file_path)
print("\n====================================")
print("DATASET CHECK COMPLETE")
print("====================================")
print("Total image files :", total_files)
print("Valid images      :", valid_files)
print("Bad files         :", len(bad_files))
if bad_files:
    print("\n❌ BAD FILES:\n")
    for file in bad_files:
        print(file)
else:
    print("\n✅ No corrupted images found!")