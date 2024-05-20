import csv
import numpy as np
from hgrlib import init_plotting
from matplotlib import pyplot as plt

init_plotting()

def read_csv_to_numpy(filename):
    # Open the CSV file
    with open(filename, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        
        # Extract the header (first row)
        header = next(csvreader)
        
        # Initialize a dictionary to hold columns
        columns = {col: [] for col in header}
        
        # Iterate through the remaining rows
        for row in csvreader:
            for col, value in zip(header, row):
                    columns[col].append(float(value))
    
    # Convert lists to numpy arrays
    for col in columns:
        columns[col] = np.array(columns[col], dtype=np.float64)  # Assuming all data can be converted to float
    
    return columns

data=read_csv_to_numpy("output.csv")


sample_size = np.arange(10, 100, 2)
scale = 1/np.sqrt(sample_size) 

plt.scatter(sample_size, data["bests_hmu_0"])
plt.scatter(sample_size, data["lows_hmu_0"])
plt.scatter(sample_size, data["highs_hmu_0"])
plt.plot(sample_size, scale)
plt.yscale("log")
current_ylim = plt.gca().get_ylim()  # Get the current y-axis limits
plt.ylim(current_ylim[0], 1.0)       # Set only the upper limit to 1.0
plt.xlim(10, 100)

plt.figure()
plt.scatter(sample_size, data["bests_hsig_0"])
plt.scatter(sample_size, data["lows_hsig_0"])
plt.scatter(sample_size, data["highs_hsig_0"])
plt.yscale("log")
current_ylim = plt.gca().get_ylim()  # Get the current y-axis limits
plt.ylim(current_ylim[0], 1.0)       # Set only the upper limit to 1.0
plt.xlim(10, 100)

plt.show()
