import pandas as pd 


import sys


f = float(sys.argv[1])
print(float(f))

ERGFILE = "untouched_labels_crossvalid.txt"

df_ergs = pd.read_csv(ERGFILE, delimiter="\t", header=None)
print(df_ergs)
df_filtered = df_ergs[df_ergs.iloc[:,2]>f]

df_filtered.to_csv("filtered_crossvalid.txt", sep="\t", index=False, header=False)

print(df_filtered)