import base64

import pandas as pd
import tensorflow as tf
import numpy as np
from io import BytesIO



skin_type_model = tf.keras.models.load_model('saved_model/my_model')
acne_model = tf.keras.models.load_model('saved_model/acne_model')

skin_type_class_names = ['dry','normal','oily']
acne_class_names = ['0','1','2']
def load_and_prep_image(img, img_shape=224):

    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    # Rescale the image (get all values between 0 and 1)
    img = img / 255.
    return img

def classify_img(model, img, class_names=skin_type_class_names):
    """
    Imports an image located at filename, makes a prediction with model
    and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(img)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    print("Pred: ",pred)
    # Add in logic for multi-class & get pred_class name
    # if len(pred[0]) > 1:
    #     pred_class = class_names[tf.argmax(pred[0])]
    # else:
    #     pred_class = class_names[int(tf.round(pred[0]))]
    print("PredArgMax: ",int(tf.argmax(pred[0]).numpy()))
    pred_class = class_names[int(tf.argmax(pred[0]).numpy())]


    # print('Prediction Probabilities : ', pred[0])
    return pred_class


def recommend_products(answers,img_data,output,makeup,skincare):
    # output-->
    #      0-makeup
    #      1-skincare
    #      2-both

    print("ImageData: ",len(img_data))
    # while len(img_data) % 4 != 0:
    #     img_data += '='
    img_bytes=base64.b64decode(img_data)

    skin_type= classify_img(skin_type_model, img_bytes, skin_type_class_names)
    print(skin_type)

    acne_level=classify_img(acne_model,img_bytes,acne_class_names)
    print("Acne level: ",acne_level)

    products=makeup

    if output==1:
        products=skincare
    elif output==2:
        products=pd.concat([makeup,skincare])

    skin_type_filtered_products= products.loc[products["skin_type"]==skin_type]
    print("Product shape: ",products.shape)
    print("SK: ",skin_type_filtered_products.shape)

    return skin_type_filtered_products

    # print(answers)
    # print(output)

    print(len(img_bytes))