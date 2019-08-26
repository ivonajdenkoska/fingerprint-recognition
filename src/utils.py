import cv2
import glob
import fprmodules.enhancement as fe
from sklearn.model_selection import train_test_split
from constants import IMAGES_PATH


def read_images():
    '''
    Reads all images from IMAGES_PATH and sorts them
    :return: sorted file names
    '''
    file_names = [img for img in glob.glob(IMAGES_PATH + "/*.tif")]
    file_names.sort()
    return file_names


def get_image_label(filename):
    image = filename.split('/')
    return image[len(image)-1]


def get_image_class(filename):
    return get_image_label(filename).split('_')[0]


# Splits the dataset on training and testing set
def split_dataset(data, test_size):
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    return train, test


def grayscale_image(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# Enhancement using orientation/frequency filtering - Gabor filterbank
def enhance_image(image):
    img_e, mask1, orientim1, freqim1 = fe.image_enhance(image)
    return cv2.normalize(img_e, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=0)


def prepare_dataset(file_names):
    '''
    Coversion to grayscale and enhancement. Split into training and test set.
    :param file_names: All fingerprint images as file names
    :return: train_set, test_set: 2 dictionaries for training and test,
             where the key is the name of the image and the value is the image itself
    '''
    train_set = {}
    test_set = {}
    data = []  # list of tuples
    temp_label = get_image_class(file_names[0])  # sets the image class (101)

    for filename in file_names:
        img = cv2.imread(filename)
        gray_img = grayscale_image(img)
        img = enhance_image(gray_img)
        label = get_image_label(filename)
        if temp_label != get_image_class(filename):
            train, test = split_dataset(data, 0.2)
            train_set.update(train)
            test_set.update(test)
            temp_label = get_image_class(filename)
            data = []

        data.append((label, img))

        if filename == file_names[len(file_names) - 1]:
            train, test = split_dataset(data, 0.2)
            train_set.update(train)
            test_set.update(test)

    return train_set, test_set


