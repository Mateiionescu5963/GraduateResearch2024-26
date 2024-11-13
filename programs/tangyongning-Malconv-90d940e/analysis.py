# @author: mliones

import sys
import json

import pandas as pd

if __name__ == "__main__":
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

    if args[2] == "final":
        assert(len(analysis_log.keys()) > 0)
        # pandas dataframe table for fast sorting and display
        df = pd.DataFrame(columns = ["window", "stride", "test_set", "mal-benign-ratio", "dataset", "acc", "precision", "recall", "F1"])
        for k in analysis_log.keys():
            name_split = k[1:-5].replace("'", "").split(", ")
            name_split.pop(0)
            for i, v in enumerate(name_split):
                if i < 2:
                    name_split[i] = int(v)
                else:
                    name_split[i] = float(v)

            arr = analysis_log.get(k)
            df.loc[len(df)] = name_split + arr

        # skip 1.0 accuracy (assume methodological failure for now)
        df = df[df["acc"] < 1.0]
        # TODO add case later on in research to determine for sure

        # sift through data by altering different filters below:
        #df = df[df["mal-benign-ratio"] == 0.1]
        df = df[df["F1"] > 0.1]
        df = df[df["acc"] > df["mal-benign-ratio"]]

        #sort by Accuracy and F1 and display
        df = df.sort_values(by = ["F1", "acc"], ascending = False)
        print(df.head(len(df)))
    else:
        try:
            path = args[1]
            assert(path[-4:] == ".txt")
            name = path[path.rindex("/")+1:-4] #string splice to extract filename w/o extension from the path

            f = open(path, "r")

            # format: "Epoch accuracy is X.XXX, precision is X.XXX, Recall is X.XXX, F1 is X.XXX."
            raw = f.read()[6:].replace(" is ", ":")[:-1].split(",") # string splice magic to get the values by themselves
            results = []
            for r in raw:
                results.append(float(r[-5:]))

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
