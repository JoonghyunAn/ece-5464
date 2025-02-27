import pandas as pd
class StatsReport:
 def __init__(self):
     self.statsdf = pd.DataFrame()
     self.statsdf['stat'] = ['types','cardinality','mean','median','stdev', 'min', 'max','nzero','nques','nmissing']
     pass
 def addCol(self, label, d):
     zero_count = (d == 0).sum()
     question_count = (d == "?").sum()
     print(question_count)
     missing_count = (d ==" ").sum()
     self.statsdf[label] = [d.dtypes,d.nunique(),d.mean(),d.median(), d.std(),d.min(), d.max(),zero_count,question_count,missing_count]
 def to_string(self):
     return self.statsdf.to_string()
 def writetofile(self, file_path,data):
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

# class StatsReport referred from ECE5464 SP25 4
# writetofile function, zero_count referred from Copilot, modified by Joong Hyun An