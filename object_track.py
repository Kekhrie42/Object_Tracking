import pandas as pd
import numpy as np 

def main():
    df = pd.read_table("test.tsv")

    #Adds id column into the data frame
    df['ID'] = "None" #Initializing all ID's to None
    df = add_id_to_grain(df)
    
    df.to_csv('example.tsv', sep="\t")


'''
Function that adds id into the data frame
only if the score is above 0.35 and the class of the 
field is not a tube_tip

df: Original Data frame
'''
def add_id_to_grain(df):
    id = 0
    for index, row in df.iterrows():
        if row['score'] > 0.35 and row['class'] != 'tube_tip':
            df.at[index, 'ID'] = id
            id += 1
    return df

if __name__ == "__main__":
    main()
