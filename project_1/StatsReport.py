import pandas as pd
class StatsReport:
 def __init__(self):
     self.statsdf = pd.DataFrame()
     self.statsdf['stat'] = ['types','cardinality','mean','median','stdev', 'min', 'max','nzero','nques','nmissing']
     pass
 def addCol(self, label, d):
     zero_count = (d == 0).sum()
     #question_count = d.apply(lambda x: str(x).count('?')).sum()
     #missing_count = d.apply(lambda x: str(x).strip() == "").sum()
     question_count = (d == "?").sum()
     missing_count = (d =="").sum()
     #question_count = d.str.contains('?').sum()
     self.statsdf[label] = [d.dtypes,d.nunique(),d.mean(),d.median(), d.std(),d.min(), d.max(),zero_count,question_count,missing_count]
 def to_string(self):
     return self.statsdf.to_string()
 def writetofile(self, file_path,data):
        # Implement the method to write data to a file
        # For example, you might want to save the data to a CSV or Excel file
        # Here's a simple example for saving to a CSV file
       
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

# class StatsReport referred from ECE5464 SP25 4
# writetofile function referred from Copilot, modified by Joong Hyun An