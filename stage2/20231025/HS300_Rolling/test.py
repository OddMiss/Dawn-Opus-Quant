import pandas as pd
import os

# Get the directory of the Python script
script_dir = os.path.dirname(__file__) + '\\'
test = script_dir + 'test\\'

# Define the file path for saving the CSV file
# csv_file_path = os.path.join(script_dir, 'dff.csv')

df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})
df.to_csv(test + 'dd.csv')
# print(script_dir + '\dd.csv')