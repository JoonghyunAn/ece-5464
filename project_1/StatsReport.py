import pandas as pd
class StatsReport:
 def __init__(self):
     self.statsdf = pd.DataFrame()
     self.statsdf['stat'] = ['types','cardinality','mean', 'min', 'max']
     pass
 def addCol(self, label, d):
     self.statsdf[label] = [d.dtypes,d.nunique(),d.mean(), d.min(), d.max()]
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