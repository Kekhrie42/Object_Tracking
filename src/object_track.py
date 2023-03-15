import pandas as pd
import numpy as np 
from pollenClass import pollenGrain
from copy import deepcopy
import matplotlib.pyplot as plt

## Author: Kekhrie Tsurho



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
    df["centroid_x"] = round(df.apply(lambda row: (row["xmin"] + row["xmax"]) / 2, axis=1),8)
    df["centroid_y"] = round(df.apply(lambda row: (row["ymin"] + row["ymax"]) / 2, axis=1),8)
    return df

def add_to_array(df):
    # Creating a list of pollenGrain objects

    for index, row in df.iterrows():
        if(row['timepoint'] == 0):
            pollen = pollenGrain()
            pollen.add_position(row['centroid_x'], row['centroid_y'], row['timepoint'])
            pollen.add_class(row['class'], row['timepoint'])       
            arrayOfPollens.append(pollen)
            df.drop(index, inplace=True)


    for t in range(1, max(df['timepoint'])+1):
    # Get the rows corresponding to the current timepoint
        df_t = df[df['timepoint'] == t]
    
        # Get the positions of all pollens at the latest timepoint
        pollens = np.array([list(pollen.get_position().values())[-1] for pollen in arrayOfPollens])

        for index, row in df_t.iterrows():
            if(row['class'] != 'tube_tip'):
                # Calculate distances between pollens and the current centroid
                distances = np.linalg.norm(pollens - np.array([row['centroid_x'], row['centroid_y']]), axis=1)

                # Find the index of the closest pollen and update its position
                min_index = np.argmin(distances) 
                arrayOfPollens[min_index].add_position(row['centroid_x'], row['centroid_y'], t) #adding position to pollen
                arrayOfPollens[min_index].add_class(row['class'], t) #adding class to pollene
                df.drop(index, inplace=True)

import matplotlib.pyplot as plt

def visualize():
    # Get the timepoints in the data
    timepoints = set()
    for pollen in arrayOfPollens:
        timepoints.update(pollen.position.keys())

    # Loop over the timepoints and create a plot for each one
    for timepoint in sorted(timepoints):
        # Get the positions for this timepoint
        positions = []
        for pollen in arrayOfPollens:
            position = pollen.position.get(timepoint)
            if position is not None:
                positions.append(position)

        # Create a scatter plot of the positions
        if positions:
            xs, ys = zip(*positions)
            plt.scatter(xs, ys)

        # Set the title and axis labels for the plot
        plt.title(f"Timepoint {timepoint}")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")

        # Display the plot
        plt.show()

        # Clear the plot for the next iteration
        plt.clf()

    
    
            


def main():

    df = pd.read_table("test.tsv")

    df['ID'] = "None" #Initializing all ID's to None
    df = add_id_to_grain(df) #Adding ID's to grains

    #Getting rows and columsn from data frame
    num_rows, num_cols = df.shape
    add_centroid(df) #adding centroid to data frame
    add_to_array(df) #adding grains to array

    visualize() #Visualizing

    df.to_csv('example.tsv', sep="\t")

if __name__ == "__main__":
    main()

