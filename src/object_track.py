## Author: Kekhrie Tsurho
## Description: This Program takes in a tsv file and creates a list of pollenGrain objects and a list of tubeTip objects. 
#           It populates the distances dictionary with the closest distances between each pollenGrain and tubeTip. It then visualizes the pollenGrains and tubeTips on the corresponding images.

import pandas as pd
import numpy as np 
from pollenClass import pollenGrain
from pollenClass import tubeTip
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image
import os
import subprocess

import re
from decimal import Decimal
import math
import itertools


#----------------------Gloabal variables-------------------#
arrayOfPollens = [] #array of pollenGrain objects
arrayOfTubeTips = [] #array of tubeTip objects
# distances = {} #dictionary of distances between pollenGrains and tubeTips
#----------------------Functions----------------------------#

'''
Function: add_id_to_grain
Description: This function adds an ID to each pollen grain that has a score greater than 0.35.
Parameters: df (dataframe): Dataframe of the tsv file
'''
def add_id_to_grain(df):
    id = 0
    for index, row in df.iterrows():
        if row['score'] > 0.35 and row['class'] != 'tube_tip':
            df.at[index, 'ID'] = id
            id += 1
    return df


'''
Function: add_centroid
Description: This function adds a centroid to each row of the dataframe.
Parameters: df (dataframe): Dataframe of the tsv file
'''
def add_centroid(df):
    df["centroid_x"] = round(df.apply(lambda row: (row["xmin"] + row["xmax"]) / 2, axis=1),200)
    df["centroid_y"] = round(df.apply(lambda row: (row["ymin"] + row["ymax"]) / 2, axis=1),200)
    return df


'''
Function: add_to_array
Description: This function adds pollenGrain objects to the arrayOfPollens and tubeTip objects to the arrayOfTubeTips.
Parameters: df (dataframe): Dataframe of the tsv file
'''
def add_to_array(df):
    # Creating a list of pollenGrain objects
    pid = 0
    ##TODO: Add case for when new pollen is added.
    for index, row in df.iterrows():
        if(row['timepoint'] == 0 and row['score'] >= 0.35 and row['class'] != 'tube_tip'):
            pollen = pollenGrain(pid)
            pid+=1
            pollen.add_position(row['centroid_x'], row['centroid_y'], row['timepoint'])
            pollen.add_class(row['class'], row['timepoint'])       
            arrayOfPollens.append(pollen)
            df.drop(index, inplace=True)

        elif(row['timepoint'] == 0 and row['score'] >= 0.35 and row['class'] == 'tube_tip'):
            tube_tip = tubeTip(pid)
            pid+=1
            tube_tip.add_position(row['centroid_x'], row['centroid_y'], row['timepoint'])
            tube_tip.add_class(row['class'], row['timepoint'])       
            arrayOfTubeTips.append(tube_tip)
            df.drop(index, inplace=True)

        
    ##Second case when pollen is already in the array. 
    for t in range(1, max(df['timepoint'])+1):
        # Get the rows corresponding to the current timepoint
        df_t = df[df['timepoint'] == t]

        # Get the positions of all pollens at the latest timepoint
        pollens = np.array([list(pollen.get_position().values())[-1] for pollen in arrayOfPollens])
        tubes = np.array([list(tube.get_position().values())[-1] for tube in arrayOfTubeTips])

        for index, row in df_t.iterrows():
            if(row['class'] != 'tube_tip' and row['score'] >= 0.35):
                # Calculate distances between pollens and the current centroid
                distances = np.linalg.norm(pollens - np.array([row['centroid_x'], row['centroid_y']]), axis=1)

                # Find the index of the closest pollen and update its position
                min_index = np.argmin(distances)

                # Add new pollen object if the closest one is too far away
                if distances[min_index] > 0.01:
                    new_pollen = pollenGrain(pid)
                    pid += 1
                    new_pollen.add_position(row['centroid_x'], row['centroid_y'], t)
                    new_pollen.add_class(row['class'], t)
                    arrayOfPollens.append(new_pollen)
                else:
                    arrayOfPollens[min_index].add_position(row['centroid_x'], row['centroid_y'], t) #adding position to pollen
                    arrayOfPollens[min_index].add_class(row['class'], t) #adding class to pollene
                df.drop(index, inplace=True)

            elif(row['class'] == 'tube_tip' and row['score'] >= 0.35):
                # Calculate distances between pollens and the current centroid
                distances = np.linalg.norm(tubes - np.array([row['centroid_x'], row['centroid_y']]), axis=1)

                # Find the index of the closest pollen and update its position
                min_index = np.argmin(distances)

                # Add new pollen object if the closest one is too far away
                if distances[min_index] > 0.01:
                    new_tube_tip = tubeTip(pid)
                    pid += 1
                    new_tube_tip.add_position(row['centroid_x'], row['centroid_y'], t)
                    new_tube_tip.add_class(row['class'], t)
                    arrayOfTubeTips.append(new_tube_tip)
                else:
                    arrayOfTubeTips[min_index].add_position(row['centroid_x'], row['centroid_y'], t) #adding position to pollen
                    arrayOfTubeTips[min_index].add_class(row['class'], t) #adding class to pollene
                df.drop(index, inplace=True)


"""
    Calculates the closest distances between each pollen and its corresponding tube tip
    and stores the results in a dictionary.

    Parameters:
    arrayOfTubeTips (list): List of tube tip objects with position fields as dictionaries
    where the keys are timepoints and the values are tuples of x,y positions.
    arrayOfPollens (list): List of pollen objects with position fields as dictionaries
    where the keys are timepoints and the values are tuples of x,y positions.
"""
def calculate_distances():
   
    distances = {}

    # Loop through each tube tip
    for tube_tip in arrayOfTubeTips:
        tube_tip_id = tube_tip.get_id()

        # Get the positions of the tube tip at all timepoints it exists
        tube_tip_positions = tube_tip.get_position()

        # Loop through each pollen object and find the closest distance between
        # the pollen and the tube tip at each timepoint they both exist
        for pollen in arrayOfPollens:
            pollen_id = pollen.get_id()

            # Skip pollen objects that do not exist at the same timepoints as the tube tip
            if not set(tube_tip_positions.keys()).intersection(set(pollen.get_position().keys())):
                continue

            # Find the closest distance between the pollen and the tube tip at each timepoint they both exist
            closest_distance = None
            for timepoint in set(tube_tip_positions.keys()).intersection(set(pollen.get_position().keys())):
                # Get the positions of the pollen and tube tip at the current timepoint
                pollen_position = pollen.get_position()[timepoint]
                tube_tip_position = tube_tip_positions[timepoint]

                # Calculate the Euclidean distance between the pollen and tube tip at the current timepoint
                distance = np.linalg.norm(np.array(pollen_position) - np.array(tube_tip_position))

                # Update the closest distance if it is None or smaller than the current closest distance
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance

            # Add the closest distance to the dictionary of distances
            key = (tube_tip_id, pollen_id)
            if key not in distances:
                distances[key] = {}
            distances[key].update({timepoint: closest_distance})

    return distances


'''
Function: visualize
Description: This function visualizes the pollenGrains and tubeTips on the corresponding images.
Parameters: image_path (string): Path to the directory of images. 
            dir (string): Directory of images
'''
def visualize(image_path, dir):
    # Get the timepoints from the first pollen object
    timepoints = list(arrayOfPollens[0].get_position().keys())

    # Generate a list of colors for each pollen object
    pollen_colors = {}
    for p in arrayOfPollens:
        pollen_id = p.get_id()
        pollen_colors[pollen_id] = np.random.rand(3,)

    # Get all the image filenames in the directory and sort them by timepoint
    filenames = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg')], key=lambda x: int(re.findall(r"_t(\d{3})_", x)[0]))

    # Loop through all the timepoints and overlay the pollen positions onto the corresponding image
    for t in timepoints:
        # Initialize the x and y coordinate lists and color list for this timepoint
        x_coords = []
        y_coords = []
        colors = []

        # Collect the x and y coordinates and colors for each pollen object at this timepoint
        for p in arrayOfPollens:
            if t in p.get_position():
                x, y = p.get_position()[t]

                # Adjust the coordinates to fit the image dimensions
                x = Decimal(x * 2048)
                y = Decimal(y * 2048)
                
                x_coords.append(x)
                y_coords.append(y)
                colors.append(pollen_colors[p.get_id()])

        for tubes in arrayOfTubeTips:
            if t in tubes.get_position():
                x, y = tubes.get_position()[t]

                # Adjust the coordinates to fit the image dimensions
                x = Decimal(x * 2048)
                y = Decimal(y * 2048)
                
                x_coords.append(x)
                y_coords.append(y)
                colors.append(pollen_colors[p.get_id()])

        # Load the corresponding image and plot the pollen positions
        image_filename = [f for f in filenames if f"_t{t:03d}_" in f]
        if not image_filename:
            continue  # Skip this timepoint if the corresponding image is not found

        image_filepath = os.path.join(image_path, image_filename[0])
        image = Image.open(image_filepath)
        plt.imshow(image)
        plt.scatter(x_coords, y_coords, c=colors, marker='.')
        plt.title(f'Timepoint {t}')
        plt.savefig(f'timepoint_{t}.png')
        plt.show()


def main():
    print("Input the tsv file path below:")
    tsv_file = input()
    df = pd.read_table(tsv_file) #Set the tsv file to a dataframe e.g "2022-01-06_run1_34C_A4_t082_stab_predictions.tsv"

    print("Input the tsv file path below:")
    dir = input() #Set the path of the directory of inference images e.g "/2022-01-06_run1_34C_A4"
    #dir = "/2022-01-06_run1_34C_A4" #Directory of images uncomment if you want to hardcode the directory
    

    df['ID'] = "None" #Initializing all ID's to None
    df = add_id_to_grain(df) #Adding ID's to grains

    add_centroid(df) #adding centroid to data frame
    add_to_array(df) #adding grains to array

    
    path = os.getcwd() + dir

    distances = calculate_distances()

    visualize(path, dir) #Visualizing
    
    
    df.to_csv('example.tsv', sep="\t") #Output the final tsv into an output

if __name__ == "__main__":
    main()

