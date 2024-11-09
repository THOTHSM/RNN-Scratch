from RNN import *
import numpy as np
import matplotlib.pyplot as plt

class RNNTrainer:
    def __init__(self, x, y, units=200, optimizer_type="Adam", learning_rate=0.0001, 
                 beta_1=0.95, beta_2=0.999, epsilon=1e-7, decay=0.0001):
        self.rnn = RNN(units, x)  
        self.y = y
        self.loss_list = []
        
        if optimizer_type == "Adam":
            self.optimizer = Adam_Optimizer(learning_rate=learning_rate,
                                             beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
        elif optimizer_type == "SGD":
            self.optimizer = SGD_Optimizer(learning_rate=learning_rate, decay=decay,momentum=0.99)
        else:
            raise ValueError("Unsupported optimizer type. Choose 'Adam' or 'SGD'.")

    def train(self, epochs=10001):
        for epoch in range(epochs):
            self.rnn.forward(Tanh)  
            y_hat = self.rnn.y_pred
            t = self.rnn.T

            dy = y_hat - self.y
            L = np.round(0.5 * np.dot(dy.T, dy) / t, 4)
            self.loss_list.append(L)

            if epoch % 1000 == 0:
                print("-" * 25)
                print(f' Epoch {epoch}, Loss: {np.round(L[0][0],3)}')

            self.rnn.backward(dy)

            self.optimizer.pre_update()
            self.optimizer.parameter_update(self.rnn)
            self.optimizer.post_update()

        print("#" * 100)
        print(f"Min loss achieved: {min(self.loss_list)}")
        print('#' * 100)

    def plot_results(self):
        x = self.rnn.x_data
        y_hat = self.rnn.y_pred
        plt.plot(x, self.y, label="Target")
        plt.plot(x, y_hat, label="Predicted")
        plt.legend()
        plt.show()
