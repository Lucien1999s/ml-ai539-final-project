import os
import pandas as pd

CSV_PATH = "Botanist_Training_Set.csv"
IMAGE_DIR = "TrainFiles"

df = pd.read_csv(CSV_PATH)

print("CSV shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

df["FileName"] = df["FileName"].astype(str)
df["Label"] = df["Label"].astype(int)

print("\nLabel min:", df["Label"].min())
print("Label max:", df["Label"].max())

def find_image_path(image_dir, file_name):
    file_name = str(file_name)

    possible_names = [
        file_name,
        file_name + ".jpg",
        file_name + ".jpeg",
        file_name + ".png",
        file_name + ".JPG",
        file_name + ".JPEG",
        file_name + ".PNG",
    ]

    for name in possible_names:
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            return path

    return None

missing = []

for file_name in df["FileName"]:
    if find_image_path(IMAGE_DIR, file_name) is None:
        missing.append(file_name)

print("\nMissing image count:", len(missing))

if len(missing) > 0:
    print("First 20 missing:")
    print(missing[:20])
else:
    print("All images found.")