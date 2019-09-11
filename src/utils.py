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


def get_genuine_impostor_scores(all_scores, identical):
    '''
    Returns two arrays with the genuine and impostor scores.
    The genuine match scores are obtained by matching feature sets
    of the same class (same person) and the impostor match scores are obtained
    by matching feature sets of different classes (different persons)
    '''
    genuine_scores = []
    impostor_scores = []
    for i in range(0, len(all_scores)):
        if identical[i] == 1:
            genuine_scores.append(all_scores[i][1])
        else:
            impostor_scores.append(all_scores[i][1])

    return genuine_scores, impostor_scores




