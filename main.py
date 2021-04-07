import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from seaborn import heatmap
import os.path as path
from collections import Counter
import download_process_data

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.random import set_seed

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, roc_curve, accuracy_score, classification_report

rand_seed = 23
seed(rand_seed)
set_seed(rand_seed)

def data_split(data, labels, train_ratio=0.5, rand_seed=42):
    """Splits data into train, validation and test set

    Parameters
    ----------
    data : list, array
        Data elements
    labels : list, array
        Corresponding labels of data elements
    train_ratio : int, float, default=0.5
       Proportion of the dataset (between 0 and 1) to include in the train split. Remaining samples will be equally splitted between validation and test sets.
    rnd_seed : int, default=42
        Seed for reproducible output
    Returns
    -------
    x_train, x_val, x_test, y_train, y_val, y_test
        data (x_*) and labels (y_*) splits

    """

    assert 0 <= train_ratio <= 1, "Error: training set ratio must be between 0 and 1"

    x_train, x_temp, y_train, y_temp = train_test_split(data,
                                                        labels,
                                                        train_size=train_ratio,
                                                        random_state=rand_seed)

    x_val, x_test, y_val, y_test = train_test_split(x_temp,
                                                    y_temp,
                                                    train_size=0.5,
                                                    random_state=rand_seed)

    return x_train, x_val, x_test, y_train, y_val, y_test


def data_split_summary(y_train, y_val, y_test, base_path):

    dataset_count = pd.DataFrame([Counter(y_train), Counter(y_val), Counter(y_test)],
                                    index=["train", "val", "test"])
    dataset_count.plot(kind='bar')
    plt.xticks(rotation=0)
    plt.savefig(path.join(base_path, "split_summary.png"))

    print(f"Total number of samples: \n{dataset_count.sum(axis=0).sum()}")
    print(f"Class/Samples: \n{dataset_count.sum(axis=0)}")
    print(f"Split/Class/Samples: \n{dataset_count}")


def plot_metrics(history):

    plt.figure()
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid('on')
    plt.savefig('results/Accuracy.png')
    plt.clf()

    plt.figure()
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid('on')
    plt.savefig('results/Loss.png')
    plt.clf()


def plot_confusion_matrix(y_true, y_pred, labels):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    heatmap(cm, annot=True, fmt="d", cmap="rocket_r", xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"results/Confusion_Matrix.png")
    plt.clf()


def plot_roc_curve(y_true, predictions):

    fpr, tpr, thresholds = roc_curve(y_true, predictions)
    auc_value = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="ROC curve (AUC = {0:.2f})".format(auc_value))
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'r--', label='Chance (AUC = 0.50)', alpha=.8)
    plt.legend()
    plt.grid('on')
    plt.savefig(f"results/ROC_curve.png")
    plt.clf()


def plot_acc_threshold(y_true, predictions):

    thresholds = np.arange(0, 1, 0.01)
    accs = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        accs[i] = accuracy_score(y_true, predictions > threshold)

    x = thresholds[np.argmax(accs)]
    y = accs.max()

    plt.figure()
    plt.plot(thresholds, accs)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.annotate(f'max acc: ({x},{y:.2f})', xy=(x, y), xytext=(
        x, y+0.2), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.grid('on')
    plt.savefig(f"results/Accuracy_threshold.png")
    plt.clf()


if __name__ == "__main__":

    ############## Load and transform data ##############
    dataset = pd.read_csv('data/processed/dataset.csv')
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(dataset['name'])
    word_length = dataset['name'].apply(len).max()
    char_index = tokenizer.texts_to_sequences(dataset['name'])
    char_index = pad_sequences(char_index, maxlen=word_length, padding="post")
    x = to_categorical(char_index)  # onehot encoding
    y = np.array(dataset['tolkien'])

    x_train, x_val, x_test, y_train, y_val, y_test = data_split(x, y, train_ratio=0.6)
    data_split_summary(y_train, y_val, y_test, base_path='data/')

    ############## Build LSTM model ##############
    model = Sequential()
    model.add(LSTM(8, return_sequences=False,
                input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    model.summary()

    model.compile(loss="binary_crossentropy",
                optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=20)
    mc = ModelCheckpoint("best_model.h5", monitor='val_loss',
                        verbose=1, save_best_only=True)

    ############## Training ##############
    history = model.fit(x_train, y_train, batch_size=32, epochs=100,
                        validation_data=(x_val, y_val), callbacks=[es, mc])

    val_loss_per_epoch = history.history['val_loss']
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print(f"Best epoch: {best_epoch}")
    plot_metrics(history)

    ############## Evaluation ##############
    model = load_model("best_model.h5")
    metrics = model.evaluate(x=x_test, y=y_test)

    predictions = model.predict(x_test)
    threshold = 0.5
    y_pred = predictions > threshold
    plot_confusion_matrix(y_test, y_pred, labels=['Drug', 'Tolkien'])

    print(classification_report(y_test, y_pred, target_names=['Drug', 'Tolkien']))
    plot_roc_curve(y_test, predictions)
    plot_acc_threshold(y_test, predictions)