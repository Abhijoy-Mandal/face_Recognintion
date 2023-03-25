from tensorflow.keras.preprocessing import image
import numpy as np
from matplotlib.pyplot import imshow
import tensorflow as tf

def pred(model, path):
    img_path = path
    img = image.load_img(img_path, target_size=(160, 160))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = x / 255.0
    print('Input image shape:', x.shape)
    imshow(img)
    prediction = model(x)
    out = ""
    print("Class prediction vector [p(0), p(1)] = ", prediction)
    if np.max(prediction) < 0:
        out = "Abhijoy"
    else:
        out = "Not Abhijoy"

    print("Class:", out)

def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

if __name__ == "__main__":
    model = load_model("C:/Users/Dell/Downloads/abhijoy_model")
    exit = 1
    while(exit):
        img_path = input("Enter image file path")
        if img_path == '0':
            exit = 0
            continue
        pred(model, img_path)


