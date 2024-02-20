from tensorflow.keras.datasets import imdb
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.0
    return results


def review(index):
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])

    decoded_review = " ".join(
        [reverse_word_index.get(i - 3, "?") for i in train_data[index]])
    return decoded_review


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words = 10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
