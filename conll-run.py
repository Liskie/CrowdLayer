import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv1D
from keras.layers import GRU, TimeDistributed, Embedding
# from keras.layers.recurrent import GRU
# from keras.layers.wrappers import TimeDistributed
from keras.utils import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers.embeddings import Embedding
from sklearn.metrics import confusion_matrix, accuracy_score
from conlleval import conlleval

# packages for learning from crowds
from crowd_layer.crowd_layers import CrowdsClassification, MaskedMultiSequenceCrossEntropy
from crowd_layer.crowd_aggregators import CrowdsCategoricalAggregator

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


NUM_RUNS = 30
DATA_PATH = "./ner-mturk/"
EMBEDDING_DIM = 300
BATCH_SIZE = 64


embeddings_index = {}
f = open("/home/zcc/wyj/data/glove.6B/glove.6B.%dd.txt" % (EMBEDDING_DIM,))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors' % len(embeddings_index))


def read_conll(filename):
    raw = open(filename, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split(' ')
        point.append(stripped_line)
        if line == '\n':
            if len(point[:-1]) > 0:
                all_x.append(point[:-1])
            point = []
    all_x = all_x
    return all_x


all_answers = read_conll(DATA_PATH+'answers.txt')
all_mv = read_conll(DATA_PATH+'mv.txt')
all_ground_truth = read_conll(DATA_PATH+'ground_truth.txt')
all_test = read_conll(DATA_PATH+'testset.txt')
all_docs = all_ground_truth + all_test
print("Answers data size:", len(all_answers))
print("Majority voting data size:", len(all_mv))
print("Ground truth data size:", len(all_ground_truth))
print("Test data size:", len(all_test))
print("Total sequences:", len(all_docs))


X_train = [[c[0] for c in x] for x in all_answers]
y_answers = [[c[1:] for c in y] for y in all_answers]
y_mv = [[c[1] for c in y] for y in all_mv]
y_ground_truth = [[c[1] for c in y] for y in all_ground_truth]
X_test = [[c[0] for c in x] for x in all_test]
y_test = [[c[1] for c in y] for y in all_test]
X_all = [[c[0] for c in x] for x in all_docs]
y_all = [[c[1] for c in y] for y in all_docs]

N_ANNOT = len(y_answers[0][0])
print("Num annnotators:", N_ANNOT)

lengths = [len(x) for x in all_docs]
all_text = [c for x in X_all for c in x]
words = list(set(all_text))
word2ind = {word: index for index, word in enumerate(words)}
ind2word = {index: word for index, word in enumerate(words)}
labels = list(set([c for x in y_all for c in x]))
print("Labels:", labels)
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}
ind2label[0] = "O" # padding index
print('Input sequence length range: ', max(lengths), min(lengths))

max_label = max(label2ind.values()) + 1
print("Max label:", max_label)

maxlen = max([len(x) for x in X_all])
print('Maximum sequence length:', maxlen)
#%% md
# Prepare embedding matrix
#%%
num_words = len(word2ind)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2ind.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
#%% md
# Convert data to one-hot encoding
#%%
def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result
#%%
X_train_enc = [[word2ind[c] for c in x] for x in X_train]
y_ground_truth_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_ground_truth]
y_ground_truth_enc = [[encode(c, max_label) for c in ey] for ey in y_ground_truth_enc]
y_mv_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_mv]
y_mv_enc = [[encode(c, max_label) for c in ey] for ey in y_mv_enc]

y_answers_enc = []
for r in range(N_ANNOT):
    annot_answers = []
    for i in range(len(y_answers)):
        seq = []
        for j in range(len(y_answers[i])):
            #enc = -1*np.ones(max_label)
            enc = -1
            if y_answers[i][j][r] != "?":
                enc = label2ind[y_answers[i][j][r]]
            seq.append(enc)
        annot_answers.append(seq)
    y_answers_enc.append(annot_answers)

X_test_enc = [[word2ind[c] for c in x] for x in X_test]
y_test_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_test]
y_test_enc = [[encode(c, max_label) for c in ey] for ey in y_test_enc]
#%% md
# Pad sequences
#%%
# pad sequences
X_train_enc = pad_sequences(X_train_enc, maxlen=maxlen)
y_ground_truth_enc = pad_sequences(y_ground_truth_enc, maxlen=maxlen)
X_test_enc = pad_sequences(X_test_enc, maxlen=maxlen)
y_test_enc = pad_sequences(y_test_enc, maxlen=maxlen)

y_answers_enc_padded = []
for r in range(N_ANNOT):
    padded_answers = pad_sequences(y_answers_enc[r], maxlen=maxlen)
    y_answers_enc_padded.append(padded_answers)

y_answers_enc_padded = np.array(y_answers_enc_padded)
y_answers_enc = np.transpose(np.array(y_answers_enc_padded), [1, 2, 0])

n_train = len(X_train_enc)
n_test = len(X_test_enc)

print('Training and testing tensor shapes:')
print(X_train_enc.shape, X_test_enc.shape, y_ground_truth_enc.shape, y_test_enc.shape)

print("Answers shape:", y_answers_enc.shape)

N_CLASSES = len(label2ind) + 1
print("Num classes:", N_CLASSES)


# Define the base deep learning model

#Here we shall use features representation produced by the VGG16 network as the input. Our base model is then simply composed by one densely-connected layer with 128 hidden units and an output dense layer. We use 50% dropout between the two dense layers.

def build_base_model():
    base_model = Sequential()
    base_model.add(Embedding(num_words,
                        300,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=True))
    base_model.add(Conv1D(512, 5, padding="same", activation="relu"))
    base_model.add(Dropout(0.5))
    base_model.add(GRU(50, return_sequences=True))
    base_model.add(TimeDistributed(Dense(N_CLASSES, activation='softmax')))
    base_model.compile(loss='categorical_crossentropy', optimizer='adam')

    return base_model
#%% md
# Auxiliary functions for evaluating the models
#%%
def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

def eval_model(model):
    pr_test = model.predict(X_test_enc, verbose=2)
    pr_test = np.argmax(pr_test, axis=2)

    yh = y_test_enc.argmax(2)
    fyh, fpr = score(yh, pr_test)
    print('Testing accuracy:', accuracy_score(fyh, fpr))
    print('Testing confusion matrix:')
    print(confusion_matrix(fyh, fpr))

    preds_test = []
    for i in range(len(pr_test)):
        row = pr_test[i][-len(y_test[i]):]
        row[np.where(row == 0)] = 1
        preds_test.append(row)
    preds_test = [ list(map(lambda x: ind2label[x], y)) for y in preds_test]

    results_test = conlleval(preds_test, y_test, X_test, 'r_test.txt')
    print("Results for testset:", results_test)

    return results_test
#%% md
# Train the model on the true labels (ground truth) and evaluate on testset
#%%
model = build_base_model()
model.fit(X_train_enc, y_ground_truth_enc, batch_size=BATCH_SIZE, epochs=20, verbose=2)