import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing

data = pd.read_csv("Iris.csv")
data = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]]
# print(data)

encoder = preprocessing.LabelEncoder()
species = encoder.fit_transform(list(data["Species"]))
# print(species)
# print(data)
predict = "Species"
X = np.array(data.drop(predict, 1))
Y = list(species)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"ACC: {str(accuracy)}")

predicted_data = model.predict(x_test)
species_types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

for i in range(len(predicted_data)):
    print("Predicted: ", species_types[predicted_data[i]], "Actual: ", species_types[y_test[i]])
    