import os
import shutil
import csv
import sys

csv_file = "/media/pranoy/New Volume1/truesight/test.csv"
filepath = "/media/pranoy/New Volume1/True Sight/images"
new_path_prefix = "/media/pranoy/New Volume1/truesight/test"

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            print(i)
            pass    # Skip header row
        else:
            filename = row[0]
            new_filename = os.path.join(new_path_prefix, filename)
            old_filename = os.path.join(filepath, filename)
            shutil.copy(old_filename, new_filename)
print("Done!")

