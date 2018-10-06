import pandas as pd
import numpy as np

# Convert labels into numeric values
def convert_to_numeric(df):
    columns = df.columns.values

    for col in columns:
        text_vals = {}
        def convert_to_int(val):
            return text_vals[val]

        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            col_content = list(df[col].values)
            unique_elements = list(dict.fromkeys(col_content))
            i = 0
            for element in unique_elements:
                if element not in text_vals:
                    text_vals[element] = i
                    i += 1

            df[col] = list(map(convert_to_int, df[col])) 

    return df  

# Replace NaN values with the most commom class value 
def replace_NaN(df):
    columns = df.columns.values
    values = {}
    for col in columns:
        values[col] = df[col].value_counts().idxmax()
    df.fillna(value=values, inplace=True)
    return df

def main():
    df = pd.read_csv('mushroom.csv')    
    df = df.sample(n=1000)
    df['Class'] = df['Class'].map({'e': 0, 'p': 1})
    df.replace('?', np.nan, inplace=True)
    df.fillna(df['stalk-root'].value_counts().index[0], inplace=True)
    df = convert_to_numeric(df)
    df.to_csv('resample_numeric.csv')

if __name__ == "__main__":
    main()

