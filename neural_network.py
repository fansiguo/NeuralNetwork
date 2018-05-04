#!/usr/bin/env python3
#===============================================================================
#
#         FILE: neuralNetwork.py
#
#        USAGE: ./neuralNetwork.py
#
#  DESCRIPTION: 神经网络实现
#
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
#       AUTHOR: fansiguo
#      COMPANY: jd.com
#      VERSION: 1.0
#      CREATED: 2018/5/3 10:45
#     REVIEWER: 
#     REVISION: ---
#    SRC_TABLE: 
#         
#    TGT_TABLE: 
#===============================================================================
import numpy
import scipy.special
import glob
import scipy.misc
import matplotlib.pyplot as plt

class neuralNetwork :
    # initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices,wih and who
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #learning rate
        self.lr = learningrate

        # activation function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # train the neural network
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # output layer error
        output_errors = targets - final_outputs
        # hidden layer error
        hidden_errors = numpy.dot(self.who.T,output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
        pass

    # query the neural network
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

def main():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.3
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

    training_data_file = open('C:\\Users\\fansiguo\\Desktop\\data\\mnist_train.csv','r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 10
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs,targets)

    # test neural network
    # test_data_file = open('C:\\Users\\fansiguo\\Desktop\\data\\mnist_test.csv','r')
    # test_data_list = test_data_file.readlines()
    # test_data_file.close()
    # all_values = test_data_list[0].split(',')

    # scoreard = []
    #
    # for record in test_data_list:
    #     all_values = record.split(',')
    #     correct_label = int(all_values[0])
    #     print(correct_label,"correct lable")
    #
    #     inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #     outputs = n.query(inputs)
    #     label = numpy.argmax(outputs)
    #     print(label,"network's answer")
    #
    #     if(label == correct_label):
    #         scoreard.append(1)
    #     else:
    #         scoreard.append(0)
    #         pass
    # scoreard_array = numpy.asarray(scoreard)
    # print('performance = ',scoreard_array.sum()/scoreard_array.size)
    # pass

    #手写数字机器识别
    our_own_dataset = []
    for image_file_name in glob.glob('C:\\Users\\fansiguo\\Desktop\\data\\img\\?.png'):
        print('loading ...',image_file_name)
        label = int(image_file_name[-5:-4])
        img_array = scipy.misc.imread(image_file_name,flatten=True)
        img_data = 255.0 - img_array.reshape(784)
        img_data = (img_data / 255.0 * 0.99) + 0.01
        print(numpy.min(img_data))
        print(numpy.max(img_data))
        record = numpy.append(label,img_data)
        our_own_dataset.append(record)
        pass

    item = 1

    plt.imshow(our_own_dataset[item][1:].reshape(28,28),cmap='Greys',interpolation='None')

    correct_label = our_own_dataset[item][0]
    inputs = our_own_dataset[item][1:]

    outputs = n.query(inputs)
    print(outputs)
    label = numpy.argmax(outputs)
    print("network says:",label)
    if(label == correct_label):
        print("match")
    else:
        print("no match")
        pass

    plt.show()
if __name__ == '__main__':
    main()