import pandas as pd
import numpy as np 

def main():
    df = pd.read_table("test.tsv")

    #Adds id column into the data frame
    df['ID'] = "None"
    df = add_id(df)
    
    df.to_csv('example.tsv', sep="\t")


'''
Function that adds id into the data frame.
df: Original Data frame
'''
def add_id(df):
    id = 0
    for index, row in df.iterrows():
        if row['score'] > 0.35:
            df.at[index, 'ID'] = id
            id += 1
    return df

if __name__ == "__main__":
    main()
