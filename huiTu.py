import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
data = pd.read_csv('./Data/Five_year_2019_01_01-17_h_data.csv')

# Get data from the second and third columns
column2 = data.iloc[:, 1]
column3 = data.iloc[:, 2]

# Plot the data
plt.plot(column2, label='Column 2', color='blue')
plt.plot(column3, label='Column 3', color='red')

# Add labels and title
plt.xlabel('Row Index')
plt.ylabel('Value')
plt.title('Plot of Column 2 and Column 3')
plt.legend()

# Display the plot
plt.show()
