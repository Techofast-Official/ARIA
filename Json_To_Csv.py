import json
import csv
import os

def json_to_csv(json_data, csv_file):
    """
    Converts JSON data to CSV format. Appends to existing file if it exists, 
    otherwise creates a new file.

    Args:
        json_data: The JSON data to be converted.
        csv_file: The path to the CSV file.
    """

    mode = 'a' if os.path.exists(csv_file) else 'w'

    with open(csv_file, mode, newline='') as csvfile:
        if json_data: 
            fieldnames = json_data.keys() 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if the file is newly created
            if mode == 'w':
                writer.writeheader()

            writer.writerow(json_data)
        else:
            # Handle empty dictionary
            writer = csv.writer(csvfile)
            writer.writerow([]) 