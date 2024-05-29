from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('organ_classifier_model.h5')

image_height, image_width = 150, 150

def predict_class(image_path):
    img = image.load_img(image_path, target_size=(image_height, image_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)
    prediction_label = "Normal" if result[0][0] > 0.5 else "Abnormal"

    print(result)

    plt.figure(figsize=(6, 6))
    plt.imshow(image.load_img(image_path))
    plt.title(prediction_label)
    plt.axis('off')
    plt.show()

    return prediction_label

new_image_path = 'A309360_1_En_11_Fig9_HTML.jpg'

prediction = predict_class(new_image_path)
print("Prediction:", prediction)