import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential


posture_data = pd.read_csv("./data/posture.csv")

posture_data.drop(columns="index", inplace=True)


features, labels = posture_data[["Ax1", "Ay1", "Az1", "Ax2", "Ay2", "Az2"]].values, posture_data["label"].values

num_classes = len(set(labels))
labels = to_categorical(labels, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

model = Sequential()

model.add(Dense(units=32, activation="relu", input_dim=6))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=6, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

prediction = model.evaluate(x_test, y_test)

print(f"Accuracy : {prediction[1]}")

model.save("posture_predictor.h5")