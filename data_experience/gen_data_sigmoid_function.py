import random
import math
import csv

# Set up the CSV file for writing
with open("mydata.csv", "w", newline='') as myWriter:
    csvWriter = csv.writer(myWriter)

    # Write the header (column names)
    csvWriter.writerow(["x1", "x2", "class"])

    # Generate 1000 data points
    for i in range(1000):
        x1 = random.random()
        x2 = random.random()
        model = 10 * x1 - 10 * x2 + 5

        # Sigmoid function
        probability = 1.0 / (1.0 + math.exp(-model))

        # Assign class based on probability
        label = 1 if random.random() < probability else 0

        # Write the data point
        csvWriter.writerow([x1, x2, label])