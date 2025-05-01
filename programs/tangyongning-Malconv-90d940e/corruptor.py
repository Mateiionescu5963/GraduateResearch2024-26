import warnings
import sys, os
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    if len(sys.argv) == 2:
        if os.path.exists("./ds_tst.csv"):
            os.system("cp ./ds_tst.csv ./ds_tst_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv")
        
        label_table = pd.read_csv('../../data/data.csv', header=None, index_col=0).rename(columns={1: 'ground_truth'}).groupby(level=0).last()
        corrupt = label_table.sample(frac=float(sys.argv[1]))
        corrupt.to_csv("./corruption.csv")
        corrupt.to_csv("./corruption_"+sys.argv[1]+"_.csv")
