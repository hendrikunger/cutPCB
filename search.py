import glob
import pandas as pd

directory = "./../"
pathname = directory + "/**/*.ngt"
files = glob.glob(pathname, recursive=True)
for file in files:
    #print(file)
    col_names = ["index", "X1", "Y1", "Klasse", "Flaeche", "X2", "X3", "Y2", "Y3", "F1", "F2"]
    df=pd.read_csv(file, sep=',',index_col=0, header=None, names=col_names, skiprows=2) 
    if df.F2[df.F2.isin([-1])].empty:
        print("No -1 in F2")

    # if df.F2.isin([1]):
    #     print("hier")