import pandas as pd

# Open the animal classes file and convert to a dataframe
animals = pd.read_csv("animal_classes.csv")
animals_df = pd.DataFrame(animals)

# Open the training data and convert to a dataframe
animals_train = pd.read_csv("animals_train.csv")
animals_train_df = pd.DataFrame(animals_train)

train_data = animals_train_df.drop(["class_number"], axis=1)
train_target = animals_train_df.class_number


# Open the test data and convert to a dataframe
animals_test = pd.read_csv("animals_test.csv")
animals_test_df = pd.DataFrame(animals_test)

test_data = animals_test_df.drop(["animal_name"], axis=1)

# Using the KNeighbors Classifer
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=train_data, y=train_target)

predicted = knn.predict(X=test_data)

predicted = [animals_df.Class_Type[x - 1] for x in predicted]

# Open the CSV file
csv_file = open("predictions_file.csv", "w")

# Write the field headers
csv_file.write("animal_name,prediction\n")

# Write the animal name and the predicted animal class
for a, p in zip(animals_test_df.animal_name.to_list(), predicted):
    csv_file.write(str(a) + "," + str(p))
    csv_file.write("\n")

# Close the file
csv_file.close()
