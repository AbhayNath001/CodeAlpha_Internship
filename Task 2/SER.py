import os
import numpy as np
import librosa
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

data_dir = "Data"
emotions = ["happy", "sad", "angry"]

def extract_features_fixed_length(file_path, max_length=100):
    audio, sr = librosa.load(file_path, sr=None, duration=max_length)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs

def pad_or_truncate_features(features, max_length):
    if len(features[1]) < max_length:
        padding = max_length - len(features[1])
        features = np.pad(features, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    elif len(features[1]) > max_length:
        features = features[:, :max_length]
    return features

X, y = [], []
for emotion in emotions:
    emotion_dir = os.path.join(data_dir, emotion)
    for filename in os.listdir(emotion_dir):
        file_path = os.path.join(emotion_dir, filename)
        features = extract_features_fixed_length(file_path)
        X.append(features)
        y.append(emotion)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

max_length = max(max(len(features[1]) for features in X_train), max(len(features[1]) for features in X_test))

X_train = [pad_or_truncate_features(features, max_length) for features in X_train]
X_test = [pad_or_truncate_features(features, max_length) for features in X_test]

X_train = np.array(X_train).reshape(len(X_train), X_train[0].shape[0], X_train[0].shape[1], 1)
X_test = np.array(X_test).reshape(len(X_test), X_test[0].shape[0], X_test[0].shape[1], 1)

model = models.Sequential([
    layers.Input(shape=X_train[0].shape),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(emotions), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
print("Listening...")
time.sleep(7)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
sample_audio_path = "Data/angry/Angry Boy.mp3"
#sample_audio_path = "Data/happy/Full masti mood happy.mp3"
#sample_audio_path = "Data/sad/Arijit singh sad song.mp3"
sample_features = extract_features_fixed_length(sample_audio_path)
sample_features = np.expand_dims(pad_or_truncate_features(sample_features, max_length), axis=0)
predicted_class = np.argmax(model.predict(sample_features))
predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
print(f"Predicted emotion: {predicted_emotion}")