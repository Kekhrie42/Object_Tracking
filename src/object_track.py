import pandas as pd
import numpy as np 
from pollenClass import pollenGrain
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image
import os
import subprocess
import cv2
import re
from decimal import Decimal


## Author: Kekhrie Tsurho
## Description: 



#----------------------Gloabal variables-------------------#
arrayOfPollens = []
#----------------------Functions----------------------------#

def add_id_to_grain(df):
    id = 0
    for index, row in df.iterrows():
        if row['score'] > 0.35 and row['class'] != 'tube_tip':
            df.at[index, 'ID'] = id
            id += 1
    return df

def add_centroid(df):
    df["centroid_x"] = round(df.apply(lambda row: (row["xmin"] + row["xmax"]) / 2, axis=1),15)
    df["centroid_y"] = round(df.apply(lambda row: (row["ymin"] + row["ymax"]) / 2, axis=1),15)
    print(df["centroid_x"])
    print(df["centroid_y"])
    return df

def add_to_array(df):
    MAX_DISTANCE = 0.1
    # Case 1: creating new pollen grains at timepoint 0
    for index, row in df.iterrows():
        if row['timepoint'] == 0 and row['score'] >= 0.35:
            pollen = pollenGrain()
            pollen.add_position(row['centroid_x'], row['centroid_y'], row['timepoint'])
            pollen.add_class(row['class'], row['timepoint'])       
            arrayOfPollens.append(pollen)
            df.drop(index, inplace=True)
    
    # Case 2: updating existing pollen grains and creating new ones at subsequent timepoints
    for t in range(1, max(df['timepoint'])+1):
        # Get the rows corresponding to the current timepoint
        df_t = df[df['timepoint'] == t]
        
        # Get the positions and classes of all pollen grains at the latest timepoint
        latest_positions = {}
        latest_classes = {}
        for pollen in arrayOfPollens:
            latest_positions[pollen] = list(pollen.get_position().values())[-1]
            latest_classes[pollen] = list(pollen.get_class().values())[-1]

        # Update positions and classes of existing pollen grains and create new ones
        for index, row in df_t.iterrows():
            if row['class'] != 'tube_tip' and row['score'] >= 0.35:
                # Calculate distances between pollen grains and the current centroid
                distances = np.linalg.norm(np.array(list(latest_positions.values())) - np.array([row['centroid_x'], row['centroid_y']]), axis=1)
                
                # Find the index of the closest pollen grain
                min_index = np.argmin(distances)
                min_distance = distances[min_index]
                
                # Check if the closest pollen grain is close enough
                if min_distance < MAX_DISTANCE:
                    closest_pollen = list(latest_positions.keys())[min_index]
                    closest_pollen.add_position(row['centroid_x'], row['centroid_y'], t)
                    closest_pollen.add_class(row['class'], t)
                else: # Create a new pollen grain if the closest pollen grain is too far away
                    new_pollen = pollenGrain()
                    new_pollen.add_position(row['centroid_x'], row['centroid_y'], t)
                    new_pollen.add_class(row['class'], t)
                    arrayOfPollens.append(new_pollen)
        
                df.drop(index, inplace=True)

# def visualize():
#     # Get the timepoints from the first pollen object
#     timepoints = list(arrayOfPollens[0].get_position().keys())   

#     # Create a scatter plot for each timepoint
#     for t in timepoints:
#         # Initialize the x and y coordinate lists for this timepoint
#         x_coords = []
#         y_coords = []

#         # Collect the x and y coordinates for each arrayOfPollens object at this timepoint
#         for p in arrayOfPollens:
#             if t in p.get_position():
#                 x, y = p.get_position()[t]
#                 x_coords.append(x)
#                 y_coords.append(y)

#         # Create a scatter plot for this timepoint
#         plt.gca().invert_yaxis()
#         plt.scatter(x_coords, y_coords)
#         plt.title(f'Timepoint {t}')
#         plt.savefig(f'timepoint_{t}.png')
#         plt.show()



def visualize(image_path):
    # Get the timepoints from the first pollen object
    timepoints = list(arrayOfPollens[0].get_position().keys())   

    # Loop through all files in the directory
    for filename in os.listdir(image_path):
        # Check if file is a PNG image
        if filename.endswith(".jpg"):
            file = os.path.join(image_path, filename)
            image = Image.open(file)

            # Get the x and y dimensions of the image
            x_dim, y_dim = image.size

            # Get the timepoint from the filename using regex
            t = int(re.findall(r"_t(\d{3})_", filename)[0])

            # Initialize the x and y coordinate lists for this timepoint
            x_coords = []
            y_coords = []

            # Collect the x and y coordinates for each arrayOfPollens object at this timepoint
            for p in arrayOfPollens:
                if t in p.get_position():
                    x, y = p.get_position()[t]

                    # Adjust the coordinates to fit the image dimensions
                    # print(x)
                    x = Decimal(x * 2000)
                    y = Decimal(y * 2000)
                    
                    x_coords.append(x)
                    y_coords.append(y)

            # Overlay the pollen positions onto the image
            print(x_coords)
            plt.imshow(image)
            plt.scatter(x_coords, y_coords, c='r', marker='.')
            plt.title(f'Timepoint {t}')
            plt.savefig(f'timepoint_{t}.png')
            plt.show()



def main():

    df = pd.read_table("test.tsv")

    df['ID'] = "None" #Initializing all ID's to None
    df = add_id_to_grain(df) #Adding ID's to grains

    #Getting rows and columsn from data frame
    num_rows, num_cols = df.shape

    add_centroid(df) #adding centroid to data frame
    add_to_array(df) #adding grains to array

    path = os.getcwd() + "/2022-01-05_run1_26C_D2"
    visualize(path) #Visualizing
    

    df.to_csv('example.tsv', sep="\t")

if __name__ == "__main__":
    main()

