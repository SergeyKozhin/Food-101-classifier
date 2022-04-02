import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tools.image_gen_extended import preprocess_input

foods = np.array(['apple_pie',
                  'baby_back_ribs',
                  'baklava',
                  'beef_carpaccio',
                  'beef_tartare',
                  'beet_salad',
                  'beignets',
                  'bibimbap',
                  'bread_pudding',
                  'breakfast_burrito',
                  'bruschetta',
                  'caesar_salad',
                  'cannoli',
                  'caprese_salad',
                  'carrot_cake',
                  'ceviche',
                  'cheese_plate',
                  'cheesecake',
                  'chicken_curry',
                  'chicken_quesadilla',
                  'chicken_wings',
                  'chocolate_cake',
                  'chocolate_mousse',
                  'churros',
                  'clam_chowder',
                  'club_sandwich',
                  'crab_cakes',
                  'creme_brulee',
                  'croque_madame',
                  'cup_cakes',
                  'deviled_eggs',
                  'donuts',
                  'dumplings',
                  'edamame',
                  'eggs_benedict',
                  'escargots',
                  'falafel',
                  'filet_mignon',
                  'fish_and_chips',
                  'foie_gras',
                  'french_fries',
                  'french_onion_soup',
                  'french_toast',
                  'fried_calamari',
                  'fried_rice',
                  'frozen_yogurt',
                  'garlic_bread',
                  'gnocchi',
                  'greek_salad',
                  'grilled_cheese_sandwich',
                  'grilled_salmon',
                  'guacamole',
                  'gyoza',
                  'hamburger',
                  'hot_and_sour_soup',
                  'hot_dog',
                  'huevos_rancheros',
                  'hummus',
                  'ice_cream',
                  'lasagna',
                  'lobster_bisque',
                  'lobster_roll_sandwich',
                  'macaroni_and_cheese',
                  'macarons',
                  'miso_soup',
                  'mussels',
                  'nachos',
                  'omelette',
                  'onion_rings',
                  'oysters',
                  'pad_thai',
                  'paella',
                  'pancakes',
                  'panna_cotta',
                  'peking_duck',
                  'pho',
                  'pizza',
                  'pork_chop',
                  'poutine',
                  'prime_rib',
                  'pulled_pork_sandwich',
                  'ramen',
                  'ravioli',
                  'red_velvet_cake',
                  'risotto',
                  'samosa',
                  'sashimi',
                  'scallops',
                  'seaweed_salad',
                  'shrimp_and_grits',
                  'spaghetti_bolognese',
                  'spaghetti_carbonara',
                  'spring_rolls',
                  'steak',
                  'strawberry_shortcake',
                  'sushi',
                  'tacos',
                  'takoyaki',
                  'tiramisu',
                  'tuna_tartare',
                  'waffles'])


def load_image(filename):
    img = image.load_img(filename, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img


with tf.device('/cpu:0'):
    model = load_model('bestmodel_101class.hdf5', compile=False)
    file = sys.argv[1]
    data = load_image(file)

    pred = model.predict(data)[0]
    pred = pred / np.sum(pred)

    ind = pred.argsort()[-5:][::-1]
    top5foods = foods[ind]
    top5probs = pred[ind]

    for food, prob in zip(top5foods, top5probs):
        print(food, prob, sep=" ")
