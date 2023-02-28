import numpy as np 
import matplotlib.pyplot as plt


i=1

for momentum in ['0.999', '0.995', '0.99', '0.9']:
    for rate in [ '1e-05', '0.0001', '0.001', '0.01', '0.1', '1.0', '5.0']:
        with np.load('D:/Documents/NeuralNetworksWilliam/curves/3_layer/{}_{}_ae_curves'.format(momentum,rate)+'_3_layer_tanh_final.npz', allow_pickle=True) as data:
            loss = data['loss_all']
            final_loss = data['final_loss']

            print(final_loss)
            plt.figure(i)
            plt.plot(loss[0][1:2000], color = 'red', label = 'GD_efective')
            plt.plot(loss[1][1:2000], color = 'yellow', label = 'HB')
            plt.plot(loss[2][1:2000], color = 'green', label = 'NAG')
            plt.legend(loc="upper right")
            plt.title('Original NN \n (momentum = {}, lr = {})'.format(momentum, rate))
            plt.savefig('D:/Documents/NeuralNetworksWilliam/curves/3_layer/figures/{}_{}_ae_curves'.format(momentum,rate)+'_3_layer_tanh.png')
            i += 1

'''
for momentum in ['0.999', '0.995', '0.99', '0.9']:
    for rate in [ '1e-05', '0.001', '0.01', '0.1', '1.0', '5.0']:
        with np.load('D:/Documents/NeuralNetworksWilliam/curves/3_layer/{}_{}_ae_curves'.format(momentum,rate)+'_3_layer_relu_final.npz', allow_pickle=True) as data:
            loss = data['loss_all']
            final_loss = data['final_loss']

            print(final_loss)
            plt.figure(i)
            plt.plot(loss[0][1:2000], color = 'red', label = 'GD_efective')
            plt.plot(loss[1][1:2000], color = 'yellow', label = 'HB')
            plt.plot(loss[2][1:2000], color = 'green', label = 'NAG')
            plt.legend(loc="upper right")
            plt.title('Relu in Hidden Layers and Sigmoid in Final \n (momentum = {}, lr = {})'.format(momentum, rate))
            plt.savefig('D:/Documents/NeuralNetworksWilliam/curves/3_layer/figures/{}_{}_ae_curves'.format(momentum,rate)+'_3_layer_relu.png')
            i += 1

for momentum in ['0.999', '0.995', '0.99', '0.9']:
    for rate in [ '1e-05', '0.001', '0.01', '0.1', '1.0', '5.0']:
        with np.load('D:/Documents/NeuralNetworksWilliam/curves/3_layer/{}_{}_ae_curves'.format(momentum,rate)+'_3_layer_leaky_final.npz', allow_pickle=True) as data:
            loss = data['loss_all']
            final_loss = data['final_loss']

            print(final_loss)
            plt.figure(i)
            plt.plot(loss[0][1:2000], color = 'red', label = 'GD_efective')
            plt.plot(loss[1][1:2000], color = 'yellow', label = 'HB')
            plt.plot(loss[2][1:2000], color = 'green', label = 'NAG')
            plt.legend(loc="upper right")
            plt.title('LeakyRelu in Hidden Layers and Sigmoid in Final \n (momentum = {}, lr = {})'.format(momentum, rate))
            plt.savefig('D:/Documents/NeuralNetworksWilliam/curves/3_layer/figures/{}_{}_ae_curves'.format(momentum,rate)+'_3_layer_leaky.png')
            i += 1

for momentum in ['0.999', '0.995', '0.99', '0.9']:
    for rate in [ '1e-05', '0.001', '0.01', '0.1', '1.0', '5.0']:
        with np.load('D:/Documents/NeuralNetworksWilliam/curves/3_layer/{}_{}_ae_curves'.format(momentum,rate)+'_3_layer_no_fsigmoid_final.npz', allow_pickle=True) as data:
            loss = data['loss_all']
            final_loss = data['final_loss']

            print(final_loss)
            plt.figure(i)
            plt.plot(loss[0][2:2000], color = 'red', label = 'GD_efective')
            plt.plot(loss[1][2:2000], color = 'yellow', label = 'HB')
            plt.plot(loss[2][2:2000], color = 'green', label = 'NAG')
            plt.legend(loc="upper right")
            plt.title('No Sigmoid in Final \n (momentum = {}, lr = {})'.format(momentum, rate))
            plt.savefig('D:/Documents/NeuralNetworksWilliam/curves/3_layer/figures/{}_{}_ae_curves'.format(momentum,rate)+'_3_layer_no_fsigmoid.png')
            i += 1


for momentum in ['0.999', '0.995', '0.99', '0.9']:
    for rate in [ '1e-05', '0.001', '0.01', '0.1', '1.0', '5.0']:
        with np.load('D:/Documents/NeuralNetworksWilliam/curves/3_layer/{}_{}_ae_curves'.format(momentum,rate)+'_3_layer_no_hsigmoid_final.npz', allow_pickle=True) as data:
            loss = data['loss_all']
            final_loss = data['final_loss']

            print(final_loss)
            plt.figure(i)
            plt.plot(loss[0][1:2000], color = 'red', label = 'GD_efective')
            plt.plot(loss[1][1:2000], color = 'yellow', label = 'HB')
            plt.plot(loss[2][1:2000], color = 'green', label = 'NAG')
            plt.legend(loc="upper right")
            plt.title('No Sigmoid in Hidden Layers \n (momentum = {}, lr = {})'.format(momentum, rate))
            plt.savefig('D:/Documents/NeuralNetworksWilliam/curves/3_layer/figures/{}_{}_ae_curves'.format(momentum,rate)+'_3_layer_no_hsigmoid.png')
            i += 1
'''