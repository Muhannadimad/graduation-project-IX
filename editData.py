# import pandas as df
# import os
# list = os.listdir("data/raw2/box2")
# c=0
# for f in list:
#     dataFrame = df.read_csv("../data/raw2/box2/"+f)
#     # dataFrame = dataFrame.drop(0)
#     df.set_option("display.max_rows", None, "display.max_columns", None)
#     dataFrame = dataFrame.drop(dataFrame.columns[[0]], axis=1)
#     dataFrame.to_csv("data/raw2/box2/c"+str(c)+".csv", header=False, index=False)
#     c+=1
#     print(c)








import pandas as pd
import os
list = os.listdir("data/raw2/box")
c=0
for f in list:
    dataFrame = pd.read_csv("data/raw2/box/"+f)
    dataFrame.columns=["0","1","2","3","4","5","6","7","8","9"]
    # dataFrame = dataFrame.drop(0)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # dataFrame = dataFrame.drop(dataFrame.columns[[0]], axis=1)
    # dataFrame = dataFrame.drop(dataFrame.columns[[0]], axis=1)
    dataFrame.to_csv("data/interim4/c"+str(c)+".csv", header=True, index=False)
    c+=1
    print(c)