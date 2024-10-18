import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  

import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

import logging
Log_Format = "%(message)s"
logging.basicConfig(filename = "logfile.log",
                    filemode = "w+",
                    format = Log_Format, 
                    level = logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logger = logging.getLogger()

np.random.seed(1)
tf.random.set_seed(1)

activities = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING',
}

features = {
    "body_acc_x": 0, 
    "body_acc_y": 1,
    "body_acc_z": 2,
    "body_gyro_x": 3,
    "body_gyro_y": 4,
    "body_gyro_z": 5,
    "total_acc_x": 6,
    "total_acc_y": 7,
    "total_acc_z": 8
}

timestep = 128
features_number = 9

lr = 0.001
epochs = 200

def main():
    for data in ['train', 'test']:
        # Convert training and testing data into numpy array
        input_features, label = convert_data_into_numpy(data)
        # Convert numpy array label to one-hot label (no order)
        label = convert_label_into_one_hot(label)
        # Load numpy array to StellarGraph object, each with nodes features and graph structures
        graphs_list = load_to_stellargraph(input_features)

        # We assume you want to generate graphs from a list, but for training,
        # you will only use the first graph. If you need to train on multiple graphs,
        # you may need to adjust this approach accordingly.
        main_graph = graphs_list[0]  # Use the first graph as an example
        generator = FullBatchNodeGenerator(main_graph)

        sample_index = np.arange(input_features.shape[0])
        # Split train and validation dataset as 80%, 20%
        split_train_valid = int(len(sample_index) * 0.8)

        if data == "train":
            train_gen = generator.flow(sample_index[:split_train_valid], targets=label[:split_train_valid])
            valid_gen = generator.flow(sample_index[split_train_valid:], targets=label[split_train_valid:])
        elif data == "test":
            test_gen = generator.flow(sample_index, targets=label)

    model = graph_classificaiton_model(generator)
    # Apply early stopping and save the best model
    es = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    mc = ModelCheckpoint('gnn_model', monitor='val_loss', mode='min', save_best_only=True)
    history = model.fit(train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, shuffle=True, callbacks=[es, mc])
    # Evaluate on testing dataset
    loss, test_acc = model.evaluate(test_gen, verbose=1)
    logger.info(f"\nLoss on testing dataset: {loss}")
    logger.info(f"Accuracy on testing dataset: {test_acc}")

    # Plot the training history
    plot(history, "acc", "accuracy")
    plot(history, "loss", "loss")

def plot(history, metrics, full_name):
    plt.plot(history.history[metrics])
    plt.plot(history.history[f'val_{metrics}'])
    plt.title(f'GNN {full_name}')
    plt.ylabel(full_name)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig(f"plot/GNN_{full_name}.jpeg")
    plt.close()

def convert_data_into_numpy(file_name):
    df = pd.DataFrame() 
    data_path = f"UCI HAR Dataset/{file_name}"

    # Labels
    label = []
    with open(f"{data_path}/y_{file_name}.txt", "r") as f:
        for line in f:
            if len(line) > 0:
                label.append(int(line) - 1)
    label = np.array(label)
    logger.info(f"Number of {file_name} samples: {len(label)}")

    # Load input features, shape: (7352, 128, 9)
    input_features = np.zeros((len(label), timestep, features_number), dtype=np.float32)
    logger.info(f"Shape of input features: {np.shape(input_features)}")

    input_features_file_path = f"{data_path}/Inertial Signals"
    for filename in os.listdir(input_features_file_path):
        with open(f"{input_features_file_path}/{filename}", "r") as f:
            for position, line in enumerate(f):
                values_list = line.split()
                assert len(values_list) == timestep
                feature_name = '_'.join(x for x in filename.split('_')[:-1])
                feature_order = features[feature_name]
                input_features[position, :, feature_order] = np.array(values_list, dtype=np.float32)

    assert len(input_features) == len(label)

    return input_features, label

def load_to_stellargraph(input_features):
    # Create graph edges
    source_node = np.arange(timestep - 1)
    target_node = np.arange(1, timestep)

    edges = pd.DataFrame({"source": source_node, "target": target_node})

    graphs_list = []
    for each_sample in range(np.shape(input_features)[0]):
        each_graph_feature_array = input_features[each_sample]

        # Create a DataFrame for node features
        node_features = pd.DataFrame(each_graph_feature_array, index=np.arange(timestep))

        each_graph = StellarGraph(node_features=node_features, edges=edges)
        graphs_list.append(each_graph)

    logger.info(f"A graph sample: {graphs_list[0].info()}")

    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_list],
        columns=["nodes", "edges"],
    )
    logger.info(f"All graphs summary: {summary.describe().round(1)}")
    return graphs_list

def convert_label_into_one_hot(label):
    one_hot_label = np.zeros((len(label), 6), dtype=np.float32)
    for position in range(len(label)):
        one_hot_label[position][label[position]] = 1
    return one_hot_label

def graph_classificaiton_model(generator):
    # Create GAT model
    gc_model = GAT(
        layer_sizes=[8, 8],
        activations=["relu", "softmax"],
        n_out=6,
    )

    # Create the input layer
    x_in = generator.node_features()  # Input layer

    # Output layer
    predictions = gc_model(x_in)

    # Create Keras model
    model = Model(inputs=x_in, outputs=predictions)

    model.compile(
        optimizer=Adam(lr),
        loss=categorical_crossentropy,
        metrics=["acc"]
    )
    return model

if __name__ == '__main__':
    main()
