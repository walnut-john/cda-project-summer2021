from numpy.core.numeric import full
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.transform import resize

from imageio import imread
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


def split_and_stratify_data(full_data, train_pct = 0.8, random = 42):
    """Function to split and stratify full dataset resulting in 
    train, test, and validation sets with equal distributions of classes.
    Just another method to try.

    Args:
        full_data (pandas df): Full dataset to split into train test and validation
        train_pct (float, optional): % of data to train on. Defaults to 0.8.
        random (int, optional): Random State. Defaults to 42.

    Returns:
        3 pandas df: train, test, validation sets
    """

    train, remaining = train_test_split(full_data, train_size = train_pct, shuffle = True, random_state = random, stratify = full_data['cancer_type'])
    test, validate = train_test_split(remaining, train_size = 0.50, shuffle = True, random_state = random, stratify = remaining['cancer_type'])
    
    '''
    print('Train Size: ', len(train.index))
    print('Test Size: ', len(test.index))
    print('Validate Size: ', len(validate.index))
    print(train.groupby(['cancer_type']).agg(['count']))
    print(test.groupby(['cancer_type']).agg(['count']))
    print(validate.groupby(['cancer_type']).agg(['count']))
    '''

    return train, test, validate


def get_balanced_df(df, val_size=350, col='label'):
    """Function to balance dataset based on Class and add Weight.
    Args:
        df (DataFrame, required): Data to split
        val_size (int, optional): Sample size for each value in class
        col (str, optional): Column for class balance
    Returns:
        DataFrame: balanced DataFrame based on class and weight.
    """
    balanced = []
    for val in df[col].unique():
        label_i = df[df[col] == val]
        if len(label_i.index) >= val_size:
            label_i = label_i.sample(val_size)
            weight_i = 1.0
        else:
            weight_i = val_size/len(label_i.index)
        label_i['class_weight'] = weight_i
        balanced.append(label_i)
        
    balanced_df = pd.concat(balanced)
    return balanced_df


def read_images(full_dataset, data_path = 'data/archive/'):
    """[summary]


    Images are 450 x 600 x 3 - which is equal to a vector of length 810,000.. Too many features. Need to downsample
    Args:
        full_dataset ([type]): [description]
        data_path (str, optional): [description]. Defaults to 'data/archive/'.
    """

    images = []

    for name, value in full_dataset['image'].iteritems():
        IMAGE_PATH = f'{data_path}images/{value}'
        img = imread(IMAGE_PATH, as_gray=False) # grey
        img = resize(img, (28,28,3))
        length = 2352 # number of pixels total (450 x 600 x 3 (if RGB))
        img = img.reshape(length)
        images.append(img)
        print(len(images))

    
    X = np.array(images)

    # PCA doesn't work with this many features.. What to do?
    '''
    pca = PCA(n_components=30, random_state=42)

    X_reduced = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)

    X_reduced = pd.DataFrame(X_reduced)
    X_reduced.to_csv('pca_features.csv', index = False)
    return X_reduced
    '''

if __name__ == "__main__":
    x = read_labels()
    train, test, validate = split_and_stratify_data(full_data = x)

    train_df2 = get_balanced_df(train, val_size = 2000, col='cancer_type')
    print(train_df2.groupby(['cancer_type','class_weight']).agg(['count']))
    X = read_images(full_dataset = x)
    print(X.shape)