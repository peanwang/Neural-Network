import numpy as np
from scipy import special 
import json

class NeuralNetwork:
    '''
    神经网络类
    '''

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        '''
        初始化函数  -- 设定输入层节点，隐藏层节点和输出层节点的数量
        parms: 
        inputnodes type: int  输入层节点数量
        hiddennodes type:int  隐藏层节点数量
        outputnodes type:int  输出层节点数量
        learningrate type:float 学习率
        '''
        
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: special.expit(x)
                   

    def train(self, inputs_list, targets_list):
        '''
        训练 -- 学习给定训练集样本后， 优化权重
        '''
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 计算信号进入隐藏层
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算信号出隐藏层
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算信号进入最终层
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算信号出最终层
        final_outputs = self.activation_function(final_inputs)       

        # error 误差函数
        output_errors = targets - final_outputs
        # 隐藏层的反向传播的误差
        hidden_errors = np.dot(self.who.T, output_errors)
        # 跟新隐藏层和最终层的权重
        self.who += self.lr * np.dot( (output_errors*final_outputs * (1.0 - final_outputs)), 
                                    np.transpose(hidden_outputs))
        # 更新输入层和隐藏层的权重
        self.wih += self.lr * np.dot( (hidden_errors*hidden_outputs * (1.0 - hidden_outputs)),
                                    np.transpose(inputs))


    def query(self, input_list):
        '''
        查询  -- 给定输入，从输出节点给出答案
        '''
        inputs = np.array(input_list, ndmin=2).T
        # 计算信号进入隐藏层
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算信号出隐藏层
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算信号进入最终层
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算信号出最终层
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    
    def save(self):
        np.save('wih', self.wih)
        np.save('who', self.who)



