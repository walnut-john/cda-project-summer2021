import glob
from os.path import exists

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.transform import resize

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def read_labels(data_path = 'data/'):
    """Function to read in the .csv file 'GroundTruth' and reverse the one-hot encoding that was applied.
    (Should be better for classification algorithms?)

    Args:
        data_path (str, optional): Path of .csv file. Defaults to 'data/'.

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


def split_and_stratify_data(full_data, train_pct = 0.8, random = 1):
    """Function to split and stratify full dataset resulting in 
    train, test, and validation sets with equal distributions of classes.

    Args:
        full_data (pandas df): Full dataset to split into train test and validation
        train_pct (float, optional): % of data to train on. Defaults to 0.8.
        random (int, optional): Random State. Defaults to 1.

    Returns:
        3 pandas df: train, test, validation sets
    """

    train, remaining = train_test_split(full_data, train_size = train_pct, shuffle = True, random_state = random, stratify = full_data['cancer_type'])
    test, validate = train_test_split(remaining, train_size = 0.50, shuffle = True, random_state = random, stratify = remaining['cancer_type'])
    
    return train, test, validate


def get_balanced_df(df, val_size = 350, col='label'):
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


def get_imgs(img_dir='data/images/*'):
    imgs = glob.glob(img_dir)
    return imgs
    
    
def read_img(img_path):
    img = plt.imread(img_path)
    #print('Image Shape: ', img.shape)
    return img
    
    
def plot_img(img):
    plt.imshow(img)
    #plt.savefig('test.png')


def reshape_img(img, scale=4):

    img_resized = resize(img, (img.shape[0] // scale, img.shape[1] // scale), anti_aliasing=True)
    length = np.prod(img_resized.shape)
    vectorized_image = img_resized.reshape(length)
    #print('Image Resized: ', img.shape, ' -> ', img_resized.shape)
    return vectorized_image, img_resized.shape 


def get_img_generator(df, img_dir=r'data/images/', x='image', y='cancer_type', h=112, w=150, batch=50):
    gen = ImageDataGenerator()
    gen_data = gen.flow_from_dataframe(df, img_dir, x_col=x, y_col=y, target_size=(h, w), batch_size=batch,
                                        class_mode='categorical') #, color_mode='rgb'
    return gen_data


def get_all_data(process_images_again = False):

    df = read_labels()
    print('GroundTruth.csv read. Processing/searching for image data. May take a few minutes if process_images_again == True.\n')

    if exists('data/image_data.csv') and process_images_again == False:
        image_data = pd.read_csv('data/image_data.csv')
        print('Existing image data read..\n')
    else:
        x_imgs = [read_img('data/images/'+img) for img in df['image']] 
        image_data = np.array([reshape_img(img, scale = 20)[0] for img in x_imgs]) # changed scale = 10 for smaller images
        image_data = pd.DataFrame(image_data)
        image_data.to_csv('data/image_data.csv', index = False) # write to .csv to avoid processing again next time
        print('Images processed and saved as a .csv file in data/ folder. \n')

    #plot_img(np.array(image_data.iloc[0,:]).reshape((22, 30, 3))) # test for images reshaped - looks good
    pca = PCA(n_components=50, random_state=42)
    image_data_reduced = pca.fit_transform(image_data)
    exp_var = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var)

    print(cum_sum_eigenvalues)
    print('\n')
    
    image_data_reduced = pd.DataFrame(image_data_reduced)

    combined_data = pd.concat((df, image_data_reduced), axis = 1)

    print(combined_data.head())

    return combined_data


if __name__ == "__main__":
    all_data = get_all_data(process_images_again = False)
    print('Done!')