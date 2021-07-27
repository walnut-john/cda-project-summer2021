import pandas as pd

# How to read in the images? Hmmm..


def read_labels(data_path = 'data/archive/'):
    """Function to read in the .csv file 'GroundTruth' and reverse the one-hot encoding that was applied.
    (Should be better for classification algorithms?)

    Args:
        data_path (str, optional): Path of .csv file. Defaults to 'data/archive/'.

    Returns:
        Pandas df: two column pandas dataframe - image name and cancer type
    """

    # read in csv file
    df = pd.read_csv(f'{data_path}GroundTruth.csv')
    df['image'] = df['image'] + '.jpg' # adds filetype to image column

    # get list of categories
    categories = df.columns.values.tolist()
    categories.remove('image')

    # reverse the one-hot encoding
    new_col = df[categories].idxmax(1)
    new_col = pd.DataFrame(new_col, columns = ['cancer_type'])
    df = pd.concat([df['image'], new_col], axis = 1)
    
    return df

if __name__ == "__main__":
    x = read_labels()
    print(x.head())