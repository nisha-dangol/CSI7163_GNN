import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  

import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GAT
from stellargraph import StellarGraph

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

# Rest of the code follows as before
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
batch_size = 100



def main():
    for data in ['train', 'test']:
        input_features, label = convert_data_into_numpy(data)
        label = convert_label_into_one_hot(label)
        graphs_list = load_to_stellargraph(input_features)
        generator = PaddedGraphGenerator(graphs=graphs_list)
        sample_index = [i for i in range(input_features.shape[0])]
        split_train_valid = int(len(sample_index) * 0.8)

        if data == "train":
            train_gen = generator.flow(sample_index[:split_train_valid], targets=label[:split_train_valid], batch_size=batch_size)
            valid_gen = generator.flow(sample_index[split_train_valid:], targets=label[split_train_valid:], batch_size=batch_size)
        elif data == "test":
            test_gen = generator.flow(sample_index, targets=label, batch_size=batch_size)

    model = graph_classificaiton_model(generator)
    
    es = EarlyStopping(monitor="val_loss", min_delta=0, patience=50, restore_best_weights=True)
    mc = ModelCheckpoint('gnn_model', monitor='val_loss', mode='min', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1, min_lr=1e-5)

    history = model.fit(train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, shuffle=True, callbacks=[es, mc, lr_scheduler])
    
    loss, test_acc = model.evaluate(test_gen, verbose=1)
    logger.info(f"\nLoss on testing dataset: {loss}")
    logger.info(f"Accuracy on testing dataset: {test_acc}")

    plot(history, "acc", "accuracy")
    plot(history, "loss", "loss")

    converter = tf.lite.TFLiteConverter.from_saved_model('gnn_model/')
    tflite_model = converter.convert()
    
    with open('gnn_model/gnn.tflite', 'wb') as f:
        f.write(tflite_model)


    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_signature_list = interpreter._get_full_signature_list() 

    logger.info(f"\nTFLite model input details:")
    for each_input_tensor in input_details:
        logger.info(f"\n{each_input_tensor}")
    logger.info(f"\nTFLite model output details: \n{output_details}")
    logger.info(f"\nTFLite model signature details: \n{input_signature_list}")

    input_shape_0 = input_details[0]['shape']
    input_data_0 = np.array(np.random.random_sample(input_shape_0), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data_0)

    input_shape_1 = input_details[1]['shape']
    input_data_1 = np.array(np.random.random_sample(input_shape_1), dtype=np.float32)
    interpreter.set_tensor(input_details[1]['index'], input_data_1)

    input_shape_2 = input_details[2]['shape']
    input_data_2 = np.array(np.random.random_sample(input_shape_2), dtype=np.bool_)
    interpreter.set_tensor(input_details[2]['index'], input_data_2)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    assert len(output_data[0]) == 6


def plot(history, metrics, full_name):
    plt.plot(history.history[metrics])
    plt.plot(history.history[f'val_{metrics}'])
    if full_name == "accuracy":
        plt.yticks(np.arange(0.2, 1.1, 0.1))
    else:
        plt.yticks(np.arange(0, 1.5, 0.2))
    plt.title(f'GNN {full_name}')
    plt.ylabel(full_name)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig(f"plot/GNN_{full_name}.jpeg")
    plt.close()


def convert_data_into_numpy(file_name):
    df = pd.DataFrame() 
    data_path = f"UCI HAR Dataset/{file_name}"

    label = []
    with open(f"{data_path}/y_{file_name}.txt", "r") as f:
        for position, line in enumerate(f):
            if len(line) > 0:
                label.append(int(line) - 1)
    label = np.array(label)
    logger.info(f"Number of {file_name} samples: {len(label)}")

    input_features = np.zeros((len(label), timestep, features_number), dtype=np.float32)
    logger.info(f"Shape of input features: {np.shape(input_features)}")

    input_features_file_path = f"{data_path}/Inertial Signals"
    for filename in os.listdir(input_features_file_path):
        with open(f"{input_features_file_path}/{filename}", "r") as f:
            for position, line in enumerate(f):
                values_list = line.split(" ")
                values_list = [float(x) for x in line.split(" ") if len(x) != 0]
                assert len(values_list) == timestep
                feature_name = '_'.join(x for x in filename.split('_')[:-1])
                feature_order = features[feature_name]
                for each_timestamp in range(timestep):
                    input_features[position][each_timestamp][feature_order] = values_list[each_timestamp]

    assert len(input_features) == len(label)

    output_input_dict = dict()
    for class_type in activities:
        output_input_dict[class_type] = dict()
        for each_feature in features:
            output_input_dict[class_type][each_feature]= []

    features_inverse = {v: k for k, v in features.items()}
    for sample_index in range(len(input_features)):
        for timestamp_index in range(len(input_features[sample_index])):
            for feature_index in range(len(input_features[sample_index][timestamp_index])):
                feature_name = features_inverse[feature_index]
                output_input_dict[label[sample_index]][feature_name].append(input_features[sample_index][timestamp_index][feature_index])

    for class_type in activities:
        for each_feature in features:
            plt.plot(np.arange(0, 50), output_input_dict[class_type][each_feature][:50], label=each_feature)
        plt.legend(title=activities[class_type], title_fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('values')
        plt.xlabel('timestamps')
        plt.savefig(f"plot/{activities[class_type]}.jpeg", bbox_inches='tight')
        plt.close()

    return input_features, label



def load_to_stellargraph(input_features):
    source_node = []
    target_node = []
    for node in range(timestep - 1):
        source_node.append(node)
        target_node.append(node + 1)
    edges = pd.DataFrame({"source": source_node, "target": target_node})

    graphs_list = []
    for each_sample in range(np.shape(input_features)[0]):
        each_graph_feature_array = input_features[each_sample]
        each_graph = StellarGraph(each_graph_feature_array, edges)
        graphs_list.append(each_graph)
    logger.info(f"A graph sample: {graphs_list[0].info()}")

    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_list],
        columns=["nodes", "edges"],
    )
    logger.info(f"\nSummary statistics of the node and edge structure: \n{summary.describe()}")

    return graphs_list



def convert_label_into_one_hot(label):
    return np.array(pd.get_dummies(label))


def graph_classificaiton_model(generator):
    # Use GAT instead of GCN
    gc_model = GAT(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        attn_heads=8,  # Use multiple attention heads
        dropout=0.3  # Increased dropout rate
    )

    x_inp, x_out = gc_model.in_out_tensors()
    
    # Modified Dense layers with BatchNormalization
    predictions = Dense(units=128, activation="relu")(x_out)
    predictions = BatchNormalization()(predictions)
    predictions = Dense(units=64, activation="relu")(predictions)
    predictions = BatchNormalization()(predictions)
    predictions = Dense(units=32, activation="relu")(predictions)
    predictions = BatchNormalization()(predictions)
    predictions = Dense(units=6, activation="softmax")(predictions)

    # create the Keras model
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(lr), loss=categorical_crossentropy, metrics=["acc"])

    model.summary(print_fn=logger.info)
    return model


if __name__ == "__main__":
    main()
