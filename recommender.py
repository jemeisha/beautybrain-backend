import base64

import pandas as pd
import tensorflow as tf
from skin_tone.skin_detection import skin_detection
from skin_tone.skin_tone_knn import skin_tone_knn
from skin_tone.skin_tone2 import detectSkinTone
import numpy as np
from io import BytesIO
from PIL import Image

skin_type_model = tf.keras.models.load_model('saved_model/skin_type2')
acne_model = tf.keras.models.load_model('saved_model/acne_model3')

skin_type_class_names = ['dry', 'normal', 'oily']
acne_class_names = ['0', '1', '2']

def png_to_jpeg(png_data):
    # Open the PNG image from the byte array
    image = Image.open(BytesIO(png_data))

    # Convert the image to RGB format (if it's not already)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Create an in-memory buffer to receive the JPEG data
    buffer = BytesIO()

    # Save the JPEG data to the buffer
    image.save(buffer, format='JPEG')

    # Return the buffer contents as a byte array
    return buffer.getvalue()
def is_png(data):
    return data[:8] == b'\x89PNG\r\n\x1a\n'
def load_and_prep_image(img, img_shape=299):
    if is_png(img):
        print("png if")
        img=png_to_jpeg(img)


    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    # Rescale the image (get all values between 0 and 1)
    img = img / 255.
    return img


# Imports an image, makes a prediction with model and plots the image with the predicted class as the title.
def classify_img(model, img, class_names=skin_type_class_names,img_shape=299):
    # Import the image and preprocess it
    img = load_and_prep_image(img,img_shape)

    # Make a prediction
    # pred = model.predict(tf.expand_dims(img, axis=0))
    pred = model.predict(img[np.newaxis, ...])

    print("Pred: ", pred)
    # Add in logic for multi-class & get pred_class name
    # if len(pred[0]) > 1:
    #     pred_class = class_names[tf.argmax(pred[0])]
    # else:
    #     pred_class = class_names[int(tf.round(pred[0]))]
    print("PredArgMax: ", int(tf.argmax(pred[0]).numpy()))
    pred_class = class_names[int(tf.argmax(pred[0]).numpy())]
    print("class",pred_class)
    # print('Prediction Probabilities : ', pred[0])
    return pred_class


def recommend_products(answers, img_data, output, makeup, skincare):
    # output-->
    #      0-makeup
    #      1-skincare
    #      2-both

    output = int(output)

    print("ImageData: ", len(img_data))
    # while len(img_data) % 4 != 0:
    #     img_data += '='
    img_bytes = base64.b64decode(img_data)

    skin_type = classify_img(skin_type_model, img_bytes, skin_type_class_names)
    print(skin_type)

    acne_level = classify_img(acne_model, img_bytes, acne_class_names,img_shape=299)
    print("Acne level: ", acne_level)

    makeup["concern2"] = ""
    makeup["concern3"] = ""

    acne_list=makeup
    if output == 1:
        acne_list = skincare
    elif output == 2:
        acne_list = pd.concat([makeup, skincare])

    if acne_level == "1" or acne_level == "2":
        print("acne if")

        acne_list= acne_list[
            acne_list["concern"].str.contains("Acne and Blemishes", case=False) |
            acne_list["concern"].str.contains("Blackheads and Whiteheads", case=False) |
            acne_list["concern2"].str.contains("Acne and Blemishes", case=False) |
            acne_list["concern2"].str.contains("Blackheads and Whiteheads", case=False) |
            acne_list["concern3"].str.contains("Acne and Blemishes", case=False) |
            acne_list["concern3"].str.contains("Blackheads and Whiteheads", case=False)
        ]
        print("Acne list shape:", acne_list.shape)
    else:
        acne_list= pd.DataFrame()
    # mean_colour_values = skin_detection(img_bytes)
    # print("Mean cv: ", mean_colour_values.shape)
    #
    # skin_tone = skin_tone_knn(mean_colour_values)
    # print("Skin tone: ", skin_tone)

    skin_tone2= detectSkinTone(img_bytes)
    print("Skin2", skin_tone2)

    makeup_filtered = makeup
    # if skin_tone == 4 or skin_tone == 5 or skin_tone == 6:
    #     makeup_filtered = makeup_filtered.loc[
    #         (makeup_filtered["skin tone"] == "fair to light") |
    #         (makeup_filtered["skin tone"] == "light to medium")
    #         ]
    # elif skin_tone == 3 or skin_tone == 2:
    #     makeup_filtered = makeup_filtered.loc[
    #         makeup_filtered["skin tone"] == "medium to dark"
    #         ]
    # elif skin_tone == 1:
    #     makeup_filtered = makeup_filtered.loc[
    #         makeup_filtered["skin tone"] == "dark to deep"
    #         ]

    if skin_tone2 == "bright":
        makeup_filtered = makeup_filtered.loc[
            makeup_filtered["skin tone"] == "fair to light"
            ]
    elif skin_tone2 == "fair":
        makeup_filtered = makeup_filtered.loc[
            makeup_filtered["skin tone"] == "light to medium"
            ]
    elif skin_tone2 == "mild":
        makeup_filtered = makeup_filtered.loc[
            makeup_filtered["skin tone"] == "medium to dark"
            ]
    elif skin_tone2 == "dark":
        makeup_filtered = makeup_filtered.loc[
            makeup_filtered["skin tone"] == "dark to deep"
            ]
    products = makeup_filtered


    print("output: ", output)
    print("output=1: ", output == 1)

    if output == 1:
        print("skincare")
        products = skincare
    elif output == 2:
        products = pd.concat([makeup_filtered, skincare])


    skin_type_filtered_products = products.loc[products["skin_type"] == skin_type]
    print("Product shape: ", products.shape)
    print("SK: ", skin_type_filtered_products.shape)

    # products based on answers
    productsAll = pd.DataFrame()
    ansMakeup = makeup.iloc[0:0, :].copy()
    ansSkincare = skincare.iloc[0:0, :].copy()

    if output == 0:
        productsAll = makeup
        ansMakeup = makeup
    elif output == 1:
        productsAll = skincare
        ansSkincare = skincare
    else:
        productsAll = pd.concat([makeup, skincare])
        ansMakeup = makeup
        ansSkincare = skincare

    productsAns1 = pd.DataFrame()

    print("Answer1: ", answers)
    if answers[0] == "oily":
        productsAns1 = productsAll.loc[productsAll["skin_type"] == "oily"]
    elif answers[0] == "normal":
        productsAns1 = productsAll.loc[productsAll["skin_type"] == "normal"]
    elif answers[0] == "dry":
        productsAns1 = productsAll.loc[productsAll["skin_type"] == "dry"]

    # print("k: ",makeup[makeup["concern"].str.contains(answers[1])])
    productsAns2 = pd.DataFrame()
    if answers[1] is not None:
        productsAns2 = pd.concat([
            ansMakeup[ansMakeup["concern"].str.contains(answers[1], case=False)],
            ansSkincare[
                ansSkincare["concern"].str.contains(answers[1], case=False) |
                ansSkincare["concern2"].str.contains(answers[1], case=False) |
                ansSkincare["concern3"].str.contains(answers[1], case=False)
                ],
        ])

    productsAns5 = pd.DataFrame()
    if answers[4] is not None:
        productsAns5 = pd.concat([
            ansMakeup[ansMakeup["concern"].str.contains(answers[4], case=False)],
            ansSkincare[
                ansSkincare["concern"].str.contains(answers[4], case=False) |
                ansSkincare["concern2"].str.contains(answers[4], case=False) |
                ansSkincare["concern3"].str.contains(answers[4], case=False)
                ],
        ])
    conc = pd.concat([productsAns1, productsAns2, productsAns5]).drop_duplicates()

    print("proans1", productsAns1.shape)
    print("proans2", productsAns2.shape)
    print("proans5", productsAns5.shape)
    print("conc", conc.shape)

    return skin_type_filtered_products, conc, acne_list

    print(len(img_bytes))
