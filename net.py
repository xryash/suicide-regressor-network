import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense


from tensorflow.python.keras.models import load_model

# load dataset
csv_name = 'master.csv'
df = pd.read_csv(csv_name)

# filter columns
items = ['country', 'year', 'sex', 'age', 'suicides_no', 'population', ' gdp_for_year ($) ', 'gdp_per_capita ($)', 'generation']
df = df.filter(items=items)

# convert categorical items to numeric
categorical_items = ['country', 'sex', 'age', 'generation']
for item in categorical_items:
    df[item] = df[item].astype("category").cat.codes


df[' gdp_for_year ($) '] = df[' gdp_for_year ($) '].apply(lambda x: int(x.replace(',', '')))



# save new clean dataset
clean_csv_name = 'kazakhstan.csv'
df.to_csv(clean_csv_name, sep=',', encoding='utf-8', index=False)

# load clean dataset
clean_df = pd.read_csv('kazakhstan.csv')



# prepare labels and features arrays
y_dataset = clean_df['suicides_no'].values
x_dataset = clean_df.drop('suicides_no', 1).values



def split_dataset(x_dataset, y_dataset, ratio):
    """Split dataset with particular ratio"""
    arr = np.arange(len(x_dataset))
    np.random.shuffle(arr)
    num_train = int(ratio * len(x_dataset))
    x_train = x_dataset[arr[0:num_train]]
    y_train = y_dataset[arr[0:num_train]]
    x_test = x_dataset[arr[num_train:x_dataset.size]]
    y_test = y_dataset[arr[num_train:x_dataset.size]]
    return x_train, x_test, y_train, y_test


# split datasets
(x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, 0.8)

def normalize(x_train, x_test):
    """Normalize train and test features"""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    return (x_train - mean) / std, (x_test - mean) / std


# normalize datasets
(x_train, x_test) = normalize(x_train, x_test)

def train_model(x_train, y_train, x_test, y_test, epochs = 100):
    """Create and train a model"""
    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse',  metrics=['mae'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=1)

    mse, mae = model.evaluate(x_test, y_test, verbose=1)

    print(mae)

    model.save('my_model.h5')
    del model

def predict(x_test, y_test):
    model = load_model('my_model.h5')


    print(x_test[20:50])

    predictions = model.predict(x_test)

    indexes = np.arange(0, 2000, 10)

    for i in indexes:
        print('{0} - {1} = {2}.'.format(int(predictions[i][0]), y_test[i], int(predictions[i][0]) - y_test[i]))

    mse, mae = model.evaluate(x_test, y_test, verbose=1)

    print(mae)

    result = [int(predictions[i][0]) for i in indexes]

    labels = [y_test[i] for i in indexes]

    plt.plot(indexes, labels, 'ro')
    plt.plot(indexes, result, 'bo')

    plt.show()


train_model(x_train, y_train, x_test, y_test, 200)
predict(x_test, y_test)