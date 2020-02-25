import neural_network
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.3

    n = neural_network.NeuralNetwork(input_nodes, hidden_nodes, 
                                    output_nodes, learning_rate)

    with open('./data/mnist_train.csv', 'r') as data_file:
        data_list = data_file.readlines()

    for record in data_list:
        all_values = record.split(',')
        scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(scaled_input, targets)
    
    with open('./data/mnist_test.csv', 'r') as test_file:
        test_list = test_file.readlines()

    scorecard = []

    for record in test_list:        
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label, "correct label")
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs =  n.query(inputs)
        label = np.argmax(outputs)
        print(label, "newwork`s answer")
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    scorecard_array = np.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)