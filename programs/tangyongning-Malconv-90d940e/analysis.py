# @author: mliones

import sys
import json
import warnings
import os
import copy

import pandas as pd
import numpy as np
from itertools import combinations as comb
from itertools import chain
import math
from datetime import datetime
import time
import importlib

def v_function(set, mode):
    assert(type(set) == list)
    if len(set) == 0:
        #print("zero")
        return 0
    elif mode.lower() == "experimental":
        #print(",", end = "", flush = True)
        dataset_test_results = pd.DataFrame()
        while dataset_test_results.empty:
            try:
                dataset_test_results = pd.read_csv("./ds_tst.csv", index_col=0)
            except pd.errors.EmptyDataError as e:
                print("collision", end="", flush = True)
                time.sleep(1)
        compliment_df = dataset_test_results.drop(set, errors = "ignore")
        df = dataset_test_results[~dataset_test_results.index.isin(compliment_df.index)]
        scores = df["Accuracies"].to_numpy() / df["Trials"].to_numpy()
        scores = scores + 1
        scores = scores / 2
        #print(scores)
        return sum(scores)
    else:
        #TODO other modes
        raise KeyError("No such mode: "+str(mode))


def shapely_value(samples, item, value_mode = "experimental"):
    shapely = 0
    n = len(samples)

    low_bound = min(n, max(1, int(math.sqrt(n))))

    for i in range(low_bound, n):
        print("/", end="", flush = True)
        subsets = list(comb(samples, i))
        for subset in subsets:
            print(".", end="", flush = True)
            S = len(subset)
            full = list(chain.from_iterable(subset))
            shapely += ((math.factorial(S) * math.factorial(n - S - 1)) / math.factorial(n)) * (v_function(list(full) + list(item), value_mode) - v_function(list(full), value_mode))

    return shapely

def extract_scores(path):
    f = open(path, "r")

    # format: "Epoch accuracy is X.XXX, precision is X.XXX, Recall is X.XXX, F1 is X.XXX."
    raw = f.read()[6:].replace(" is ", ":")[:-1].split(",")  # string splice magic to get the values by themselves
    results = []
    for r in raw:
        results.append(float(r[-5:]))

    f.close()
    return results

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pd.set_option('display.max_rows', None)
    print(".", end = "")
    args = None
    if len(sys.argv) < 3:
        exit(500)
    elif len(sys.argv) > 3: #handle an error caused by filenames containing commas which are parsed as separate arguments
        args = [sys.argv[0], sys.argv[-1]]

        filepath = ""
        for a in sys.argv:
            if a in args:
                continue
            filepath += a
            if a[-1] == ",":
                filepath += " "

        args.insert(1, filepath)
    else:
        args = sys.argv

    assert(len(args) == 3)
    f = None

    analysis_log = {}
    try:
        f = open("./analysis.txt", "r")
        analysis_log = json.loads(f.read())
        f.close()
    except FileNotFoundError:
        print("No previous analysis log: starting new")

    if args[2] == "shapctrl":
        fraction = 0.5
        try:
            fraction = int(args[1])
        except ValueError:
            fraction = 0.5

        full_set = pd.read_csv('../../data/data.csv', header=None, index_col=0).rename(columns={1: 'ground_truth'}).groupby(level=0).last().sample(frac = 1)
        corrupt = pd.read_csv("./corruption.csv", index_col=0)
        true_set = full_set.drop(corrupt, errors="ignore")

        group_count = min(10, len(true_set))
        groups = np.array_split(true_set, group_count)

        bad_boys = corrupt.sample(n = int(len(groups[0]) * fraction)+1)

        df = pd.DataFrame(columns=["Subset_ID"])
        df.set_index("Subset_ID", inplace=True)

        for n in range(group_count):
            name = "Sv" + str(n)
            df[name] = -1

            label_sets = copy.deepcopy(groups)
            label_sets[n] = label_sets[n].sample(frac=fraction) + bad_boys

            shapely = []
            print("[", end="", flush=True)
            for s in label_sets:
                print("-", end="", flush=True)
                current = []
                for si in label_sets:
                    if not si.equals(s):
                        current.append(si)
                shapely.append(shapely_value(current, s.index.to_list()))
            print("]", flush=True)




            for i, s in enumerate(label_sets):
                df.loc[i, name] = shapely[i]

        path = "./shapctrl_logs/shapely_ctrl_"+str(fraction)+"_.csv"
        df.to_csv(path)


    elif args[2] == "analyze_shapely":
        try:
            analysis_log = pd.read_csv("shapely_analysis.csv", index_col=0)
        except FileNotFoundError:
            print("Starting new shapely analysis log")
            analysis_log = pd.read_csv('../../data/data.csv', header=None, index_col=0).rename(columns={1: 'ground_truth'}).groupby(level=0).last().sample(frac = 1)
            analysis_log["Corrupt"] = False
            try:
                corruption = pd.read_csv("./corruption.csv", index_col=0)
                for index, row in corruption.iterrows():
                    analysis_log.at[index, "Corrupt"] = True
            except FileNotFoundError:
                print("No corruption file found.")
        finally:
            analysis_log["Shapely"] = 0

        path = "./shapely_logs"

        c = 0
        for filename in os.listdir(path):
            print(".", end="", flush=True)
            fpath = os.path.join(path, filename)
            if os.path.isfile(fpath):
                if fpath[-4:] == ".csv":
                    shap = pd.read_csv(fpath, index_col=0)
                    avg = (shap["Shapely_Value"].sum()) / len(shap)

                    for index, row in shap.iterrows():
                        #analysis_log.at[index, "Shapely"] += (shap.at[index, "Shapely_Value"] / avg)
                        analysis_log.at[index, "Shapely"] += shap.at[index, "Shapely_Value"]

                    c += 1

        #analysis_log["Shapely"] /= c
        analysis_log.sort_values(by = ["Shapely"], ascending = False)
        print(analysis_log)

        analysis_log.to_csv("shapely_analysis.csv")
        analysis_log.to_csv("shapely_analysis_bak_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".csv")

    elif args[2].split("_")[0] == "shapely":
        label_table = pd.read_csv('../../data/data.csv', header=None, index_col=0).rename(columns={1: 'ground_truth'}).groupby(level=0).last().sample(frac = 1)
        #create a number of subsets equal to the sqrt of the full dataset size
        n = len(label_table)
        #subset_size = min(int(np.sqrt(n)/ 10), n)
        subset_size = min(10, n)

        if len(args[2].split("_")) > 1:
            subset_size = min(int(args[2].split("_")[1]), n)
            if len(args[2].split("_")) > 2:
                label_table = label_table.sample(frac = float(args[2].split("_")[2]))

        print("Subset Count is: "+str(subset_size))

        label_sets = np.array_split(label_table, subset_size)
        shapely = []
        print("[", end = "", flush = True)
        for s in label_sets:
            print("-", end="", flush = True)
            current = []
            for si in label_sets:
                if not si.equals(s):
                    current.append(si)
            shapely.append(shapely_value(current, s.index.to_list()))
        print("]", flush = True)

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = "./shapely_logs/shapely_log_"+now+".csv"

        df = pd.DataFrame(columns=["Name", "Subset_ID", "Shapely_Value"])
        df.set_index("Name", inplace=True)
        for i, s in enumerate(label_sets):
            for sample in s.index:
                df.loc[sample] = [i, shapely[i]]

        df.sort_values(by=["Subset_ID"], ascending=False)
        df.to_csv(path)

    elif args[2] == "final":
        assert(len(analysis_log.keys()) > 0)
        # pandas dataframe table for fast sorting and display
        df = pd.DataFrame(columns = ["window", "stride", "test_set", "mal-benign-ratio", "embed", "acc", "precision", "recall", "F1"])
        for k in analysis_log.keys():
            name_split = k[1:-5].replace("'", "").split(", ")
            name_split.pop(0)
            for i, v in enumerate(name_split):
                if i < 2:
                    name_split[i] = int(v)
                elif i + 1 == 6:
                    if not "mode" in df.columns:
                        df.insert(5, "mode", "unlabeled")
                    name_split[i] = v
                else:
                    name_split[i] = float(v)

            arr = analysis_log.get(k)
            try:
                df.loc[len(df)] = name_split + arr
            except ValueError:
                name_split.append("unlabeled")
                df.loc[len(df)] = name_split + arr


        # skip 1.0 accuracy (assume methodological failure for now)
        #df = df[df["acc"] < 1.0]
        # TODO add case later on in research to determine for sure

        # sift through data by altering different filters below:
        #df = df[df["mal-benign-ratio"] <= 0.6]
        #df = df[df["F1"] > 0.1]
        #df = df[df["acc"] > df["mal-benign-ratio"]]
        #df = df[df["window"] > 500]

        #sort by Accuracy and F1 and display
        df = df.sort_values(by = ["acc", "F1"], ascending = False)
        print(df)
    else:
        try:
            path = args[1]
            assert(path[-4:] == ".txt")
            name = path[path.rindex("/")+1:-4] #string splice to extract filename w/o extension from the path

            results = extract_scores(path)

            if name not in analysis_log.keys():
                analysis_log.setdefault(name, results)
            else:
                analysis_log[name] = results

        except FileNotFoundError:
            print("404: File Not Found")
        except ValueError:
            print(end = "")
        finally:
            if f:
                f.close()
            f = open("./analysis.txt", "w")
            f.write(json.dumps(analysis_log))
            f.close()
