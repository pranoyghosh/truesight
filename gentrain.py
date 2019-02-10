import os
import shutil
import csv
import sys
 
csv_file = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/training.csv"
filepath = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/images" 
new_path_prefix = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/training"

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
