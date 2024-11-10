from RNN import *
import numpy as np
import matplotlib.pyplot as plt

class RNNTrainer:
    def __init__(self, x_data, y_data, units=500,Activation=Tanh, optimizer_type="Adam",
                  learning_rate=0.0001,beta_1=0.95, beta_2=0.999, epsilon=1e-7,
                    decay=0.0001,dt=0):
        
        x_plot = np.arange(0,max(x_data.shape))
        self.dt = dt
        if self.dt!=0:
            x = y_data[:-dt]
            y = y_data[dt:]
            self.x_plots = x_plot[dt:]
        else:
            x,y = x_data,y_data
            self.x_plots = x_plot

        self.rnn = RNN(units, x)  
        self.y = y
        self.loss_list = []
        self.Activation = Activation
        self.units = units

        if optimizer_type == "Adam":
            self.optimizer = Adam_Optimizer(learning_rate=learning_rate,
                                             beta_1=beta_1, beta_2=beta_2,
                                               epsilon=epsilon, decay=decay)
        elif optimizer_type == "SGD":
            self.optimizer = SGD_Optimizer(learning_rate=learning_rate, decay=decay,
                                           momentum=0.99)
        else:
            raise ValueError("Unsupported optimizer type. Choose 'Adam' or 'SGD'.")

    def train(self, epochs=10001):
        all_Wx = []
        all_Wh = []
        all_Wy = []
        all_bias = []

        for epoch in range(epochs):
            self.rnn.forward(Activation_func=self.Activation)  
            y_hat = self.rnn.y_pred
            t = self.rnn.T

            all_Wx.append(self.rnn.Wx)
            all_Wy.append(self.rnn.Wy)
            all_Wh.append(self.rnn.Wh)
            all_bias.append(self.rnn.bias)

            dy = y_hat - self.y
            L = np.round(0.5 * np.dot(dy.T, dy) / (t-self.dt), 3)
            self.loss_list.append(L[0][0]) 
            if epoch % 1000 == 0:
                print("-" * 25)
                print(f' Epoch {epoch}, Loss: {L[0][0]}')

            self.rnn.backward(dy)

            self.optimizer.pre_update()
            self.optimizer.parameter_update(self.rnn)
            self.optimizer.post_update()

        print("#" * 100)
        print(f"Min loss achieved: {min(self.loss_list)}")
        print('#' * 100)
        
        index = np.argmin(self.loss_list)
        self.best_Wx = all_Wx[index]
        self.best_Wy = all_Wy[index]
        self.best_Wh = all_Wh[index]
        self.best_bias = all_bias[index]


    def plot_results(self):
        x = self.rnn.x_data
        y_hat = self.rnn.y_pred
        plt.plot(self.x_plots, self.y, label="Target")
        plt.plot(self.x_plots, y_hat, label="Predicted")
        plt.legend()
        plt.show()
    
    def test(self, x_test):
        T = x_test.shape[0]
        activation = self.Activation()
        self.y_test_pred = np.zeros((T, 1))
        self.H = [np.zeros((self.units, 1)) for _ in range(T + 1)]
        
        ht = self.H[0]
    
        for t, xt in enumerate(x_test):
            xt = xt.reshape(1, 1)
            output = np.dot(self.best_Wx, xt) + np.dot(self.best_Wh, ht) + self.best_bias
            ht = activation.forward(output)
            self.H[t + 1] = ht
            self.y_test_pred[t] = np.dot(self.best_Wy, ht)
        return self.y_test_pred





