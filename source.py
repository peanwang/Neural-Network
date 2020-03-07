import neural_network
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt


def Test(hidden_nodes, learning_rate, epochs):
    '''
    测试hidden_nodes数量对神经网络的影响
    测试learning_rate对神经网络的影响
    测试epochs对神经网络的影响
    '''
    input_nodes = 784
    output_nodes = 10

    # 构造神经网络
    n = neural_network.NeuralNetwork(input_nodes, hidden_nodes, 
                                    output_nodes, learning_rate)

    # 打开训练mnist
    with open('./data/mnist_train.csv', 'r') as data_file:
        data_list = data_file.readlines()
    
    for _ in range(epochs):
        for record in data_list:
            all_values = record.split(',')
            scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(scaled_input, targets)
    
    n.save()
    # 打开测试mnist
    with open('./data/mnist_test.csv', 'r') as test_file:
        test_list = test_file.readlines()

    score = 0

    for record in test_list:        
        all_values = record.split(',')
        correct_label = int(all_values[0])
        #print(correct_label, "correct label")
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs =  n.query(inputs)
        label = np.argmax(outputs)
        #print(label, "newwork`s answer")
        if label == correct_label:
            score += 1

    return score/len(test_list)



def Test_learning_rate():
    hidden_nodes = 100
    epochs= 1

    rates = np.linspace(0.1,0.9, 17)
    result = [] 

    p = Pool(4)
    for learning_rate in rates:
        performance =p.apply_async(Test, args=(hidden_nodes, learning_rate, epochs))
        result.append(performance)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    for i in range(len(result)):
        result[i] = result[i].get()
    result = np.array(result)

    plt.rcParams['font.sans-serif']=['SimHei']
    plt.xlabel('学习率')
    plt.ylabel('性能')
    plt.title('性能与学习率')
    plt.plot(rates, result, color='r')
    max_rate = rates[result.argmax()]
    plt.scatter(max_rate, result.max(), s=100, color='c')
    plt.plot([max_rate, max_rate], [result.max(), 0], 'k--', lw=1)
    plt.plot([0, max_rate], [result.max(), result.max()], 'k--', lw=1)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    plt.show()


def Test_hidden_nodes():
    learning_rate = 0.2
    epochs= 1

    nodes = np.linspace(100, 500, 5, dtype = int)
    result = [] 

    p = Pool(4)
    for hidden_nodes in nodes:
        performance =p.apply_async(Test, args=(hidden_nodes, learning_rate, epochs))
        result.append(performance)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    for i in range(len(result)):
        result[i] = result[i].get()
    result = np.array(result)

    plt.rcParams['font.sans-serif']=['SimHei']
    plt.xlabel('隐藏层神经元数量')
    plt.ylabel('性能')
    plt.title('性能与隐藏层神经元数量')
    plt.plot(nodes, result, color='r')
    max_nodes = nodes[result.argmax()]
    plt.scatter(max_nodes, result.max(), s=100, color='c')
    plt.plot([max_nodes, max_nodes], [result.max(), 0], 'k--', lw=1)
    plt.plot([0, max_nodes], [result.max(), result.max()], 'k--', lw=1)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    plt.show()


def Test_epochs():
    learning_rate = 0.2
    hidden_nodes = 200

    epochs = np.linspace(1, 20, 20, dtype = int)
    result = [] 

    p = Pool(4)
    for epoch in epochs:
        performance =p.apply_async(Test, args=(hidden_nodes, learning_rate, epoch))
        result.append(performance)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    for i in range(len(result)):
        result[i] = result[i].get()
    result = np.array(result)

    plt.rcParams['font.sans-serif']=['SimHei']
    plt.xlabel('世代')
    plt.ylabel('性能')
    plt.title('性能与世代')
    plt.plot(epochs, result, color='r')
    max_epoch = epochs[result.argmax()]
    plt.scatter(max_epoch, result.max(), s=100, color='c')
    plt.plot([max_epoch, max_epoch], [result.max(), 0], 'k--', lw=1)
    plt.plot([0, max_epoch], [result.max(), result.max()], 'k--', lw=1)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    plt.show()


if __name__ == '__main__':
    #Test_learning_rate()
    #Test_hidden_nodes()
    #Test_epochs()
    print(Test(200, 0.2, 5))