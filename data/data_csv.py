import sys
import os
import pandas as pd

if __name__ == "__main__":
    #python3 data_csv.py data_directory_name label (0 for benign; 1 for malware)
    assert(len(sys.argv) == 3)

    path = sys.argv[1]
    label = int(sys.argv[2])
    df = pd.DataFrame(columns = ["name", "label"])

    files = os.listdir(path)

    for file in files:
        df.loc[len(df)] = [str(file), label]

    df.to_csv("temp_data.csv", index = False)