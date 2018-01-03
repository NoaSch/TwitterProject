import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# have to use latin1 even though it results in a lot of dead characters
twitterDataSet = pd.read_csv("c:\gender-classifier-DFE-791531.csv", encoding='latin1')
twitterDataSet.head()

