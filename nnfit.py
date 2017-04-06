import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def runNN(train_set, test_set, learning_rate=0.00001, layers=(50,110,4), cost_add=100000 ):
    
    print("Running Neural Network")

    # Debug Params
    display_step = 2000
    n_samples = len(train_set)
    std_of_random= 1

    # Network Size
    size_layer_1 = layers[0] # 1st layer number of features
    size_layer_2 = layers[1] # 2nd layer number of features
    size_layer_3 = layers[2]
    size_input = 6
    size_output = 1

    # variable input
    x = tf.placeholder("float", [None, size_input], name="X")
    y = tf.placeholder("float", [None, size_output], name="Y")

    # Weights
    w1 = tf.Variable(tf.truncated_normal([size_input, size_layer_1], stddev=std_of_random))
    w2 = tf.Variable(tf.truncated_normal([size_layer_1, size_layer_2], stddev=std_of_random))
    w3 = tf.Variable(tf.truncated_normal([size_layer_2, size_layer_3], stddev=std_of_random))
    w4 = tf.Variable(tf.truncated_normal([size_layer_3, size_output], stddev=std_of_random))
    
    # Biases
    b1 = tf.Variable(tf.truncated_normal([size_layer_1], stddev=std_of_random))
    b2 = tf.Variable(tf.truncated_normal([size_layer_2], stddev=std_of_random))
    b3 = tf.Variable(tf.truncated_normal([size_layer_3], stddev=std_of_random))
    b4 = tf.Variable(tf.truncated_normal([size_output], stddev=std_of_random))

    # Network Setup
    layer_1 = tf.add(tf.matmul(x, w1), b1)
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, w3), b3)
    layer_3 = tf.nn.relu(layer_3)
    pred = tf.matmul(layer_3, w4) + b4

    # Cost Function and Optimizer
    cost = tf.divide(tf.pow(tf.subtract(pred,y),2), tf.add(tf.pow(tf.subtract(pred, y), 2), cost_add))[0][0]
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Init
    init = tf.global_variables_initializer()

    # Prepare Result Containers
    accuracy_total = []
    cost_total = []
    average_cost_total = []

    # Start Session
    with tf.Session() as sess:
        sess.run(init)

        avg_cost = 0.
        this_cost = 0.

        # Training cycle
        for i in range(len(train_set)):


            X_data, Y_data = train_set[i]

            _, c, out, tobe = sess.run([optimizer, cost, pred, y], feed_dict={x: X_data, y: Y_data})

            avg_cost += c / n_samples

            cost_total.append([c, out, tobe])
            average_cost_total.append(avg_cost)


        print("Training Finished!")

        # Test model
        correct_prediction = tf.divide(tf.subtract(pred, y), y)
        log_prediction = tf.divide(tf.subtract(tf.pow(tf.cast(10.0,"float"), pred), tf.pow(tf.cast(10.0,"float"), y)), tf.pow(tf.cast(10.0,"float"),y))
        # Calculate accuracy
        accuracy = tf.cast(correct_prediction, "float")
        accuracy_log = tf.cast(log_prediction, "float")
        for t in range(len(test_set)):
            X_data, Y_data = test_set[t]
            accuracy_total.append(sess.run([accuracy, accuracy_log, pred, y, cost], feed_dict=({x: X_data, y: Y_data})))
        
        sess.close()
    
    cost_total = np.array(cost_total)
    accuracy_total = np.array(accuracy_total)
    
    return cost_total, accuracy_total

def load_data_file(filename='W5-reduced.txt'):
    print("Loading Data File")
    names = ['z','theta','lambda','time','wx','wz','val']
    dorg = pd.read_csv(filename, header=0, names=names, delimiter=' ')

    # overwrite original value with logarithmic one
    dorg['val'] = np.log10(dorg['val'])
    print("Loading Completed")
    return dorg

def prepareDataSet(df, length=10000):
    print("Generating CV Sets")
    m = df.as_matrix()
    m = np.array([np.array(x, dtype=np.float32) for x in m])
    np.random.shuffle(m)
    #s = s[:length]
    #s = pd.DataFrame(s)
    data = []
    for i in range(length):
        X_data = np.array([m[i][0:6]], dtype=np.float32)
        Y_data = [[m[i][6]]]
        data.append([X_data, Y_data]);
    print("Generating Completed")
    return data


def run_eval(cv=20):

    d = load_data_file()

    length = len(d)
    
    all_data_shuffle = [prepareDataSet(d, length) for _ in range(cv)]
    
    splitpoint = int(length*0.8)
    train_d = [d[:splitpoint] for d in all_data_shuffle]
    test_d = [d[splitpoint:] for d in all_data_shuffle]
    
    print("Starting Loop")
    acc = []
    cos = []
    for i in range(cv):
    
        cost_train, accuracy_test = runNN(train_d[i], test_d[i])
    
        cost_train = np.array(cost_train)
        cos.append(cost_train)
        accuracy_test = np.array(accuracy_test)
        acc.append(accuracy_test)

    return cos, acc

def pretty_print_all(cos, acc):
    # Make sure all are np arrays
    acc = np.array(acc)
    cos = np.array(cos)

    # print results
    print("Average Cost with Logarithmic Values")
    print(np.average(abs(acc[:,:,0])))
    print(np.average([np.std(abs(run[:,0])) for run in acc]))
    print("Average Cost with Original Values")
    print(np.average(acc[:,:,1]))
    print(np.std(acc[:,:,1]))
    print("Average Prediction")
    print(np.average(acc[:,:,2]))
    print(np.std(acc[:,:,2]))
    print("Average Label")
    print(np.average(acc[:,:,3]))
    print(np.std(acc[:,:,3]))

    # histogram of accuracy with log
    plt.subplot(121)
    limit = 1000
    err = np.array([item[0] for x in acc for item in x])
    err = err[np.where((err > -1 * limit) & (err < limit))]
    plt.hist(err, bins=100)
    plt.title('Histogram of cost with logarithmic values')
    plt.xlabel('Cost')

    # histogram of accuracy without log
    plt.subplot(122)
    limit = 100
    err1 = np.array([item[1] for x in acc for item in x])
    err = err1[np.where((err1 > -1 * limit) & (err1 < limit))]
    plt.hist(err, bins=100)
    plt.title('Histogram of cost with original values')
    plt.xlabel('Cost')
    plt.subplots_adjust(top=0.92, bottom=0.20, left=0.10, right=0.95, hspace=0.30,
                        wspace=0.3)
    plt.show()

    # histogram of accuracy without log
    limit = 1000
    err = np.array([item[2] for x in acc for item in x])
    err = err[np.where((err > -1 * limit) & (err < limit))]
    plt.hist(err, bins=100)
    #plt.xlim((-100,100))
    plt.title('Histogram of predictions with logarithmic values')
    plt.show()

    # histogram of accuracy without log
    limit = 1000
    err = np.array([item[3] for x in acc for item in x])
    err = err[np.where((err > -1 * limit) & (err < limit))]
    plt.hist(err, bins=100)
    #plt.xlim((-100,100))
    plt.title('Histogram of labels with logarithmic values')
    plt.show()

def pretty_print_single(acc, cos, i):
    cost_total = cos[i]
    accuracy_total = acc[i]

    print('Average Cost')
    print(np.average(accuracy_total[:,0]))
    print('Variance of Cost')
    print(np.std(accuracy_total[:,0]))

    plt.hist(accuracy_total[:,0], range=(-15,15), bins=100)
    plt.title('TEST_ACCURACY_LOGARITHMIC (-15 to 15)')
    plt.show()

    #plt.hist(accuracy_total[:,2], range=(-100,100), bins=40)
    #plt.hist(accuracy_total[:,1], range=(-100,100), bins=40)

    plt.hist(accuracy_total[:,1], range=(-15,15), bins=100)
    plt.title('TEST_ACCURACY_ORIGINAL (-15 to 15)')
    plt.show()

    plt.scatter(range(len(accuracy_total)), accuracy_total[:,2])
    plt.title('TEST_PREDICTIONS_LOGARITHMIC')
    plt.show()

    plt.scatter(range(len(accuracy_total)), accuracy_total[:,3])
    plt.title('TEST_LABEL_LOGARITHMIC')
    plt.show()

    plt.plot(cost_total[:,0])
    plt.title('TRAIN_COST')
    plt.show()

    plt.plot(cost_total[:,1])
    plt.title('TRAIN_PREDICTION')
    plt.ylim((0, 25))
    plt.show()

    plt.plot(cost_total[:,2])
    plt.title('TRAIN_LABEL')
    plt.show()


cos, acc = run_eval(cv=2)
pretty_print_all(cos, acc)
