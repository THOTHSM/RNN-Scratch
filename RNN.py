import numpy as np

class Tanh:
    def forward(self,inputs):
        self.inputs = inputs 
        self.outputs = np.tanh(inputs)
        return self.outputs 
    
    def backward(self,dvalues):
        np.clip(dvalues, -5, 5, out=dvalues)  
        derivative = 1-self.outputs**2
        self.dinputs = np.multiply(dvalues ,derivative)
        return self.dinputs

class RNN:
    def __init__(self,no_of_neurons,x_data):
        self.no_of_neurons = no_of_neurons
        self.x_data = x_data.reshape(-1,1)
        self.T = max(x_data.shape)

        self.y_pred = np.zeros((self.T,1))
        self.H = [np.zeros((self.no_of_neurons,1)) for i in range(self.T+1)]

        # self.Wx = 0.1*np.random.randn(self.no_of_neurons,1)
        # self.Wh = 0.1*np.random.randn(self.no_of_neurons,self.no_of_neurons)
        # self.Wy = 0.1*np.random.randn(1,self.no_of_neurons)
        self.bias = np.zeros((self.no_of_neurons,1))
        self.Wx = np.random.randn(self.no_of_neurons, 1) * np.sqrt(1. / self.no_of_neurons)
        self.Wh = np.random.randn(self.no_of_neurons, self.no_of_neurons) * np.sqrt(1. / self.no_of_neurons)
        self.Wy = np.random.randn(1, self.no_of_neurons) * np.sqrt(1. / self.no_of_neurons)


    def forward(self,Activation_func):
        self.dWx = np.zeros((self.no_of_neurons,1))
        self.dWh = np.zeros((self.no_of_neurons,self.no_of_neurons))
        self.dWy = np.zeros((1,self.no_of_neurons))
        self.dbias = np.zeros((self.no_of_neurons,1))

        self.Activation = [Activation_func() for t in range(self.T)] 
        ht = self.H[0]
        for t,xt in enumerate(self.x_data):
            xt = xt.reshape(1,1)
            output = np.dot(self.Wx,xt)+np.dot(self.Wh,ht)+self.bias
            ht  = self.Activation[t].forward(output) 
            self.H[t+1] = ht 
            self.y_pred[t] = np.dot(self.Wy,ht)

    def backward(self,dvalues):
        dht = np.zeros_like(self.H[0])
        for t in reversed(range(self.T)):
            dy = dvalues[t].reshape(1,1)
            xt = self.x_data[t].reshape(1,1)

            dht += np.dot(self.Wy.T, dy)

            dtanh = self.Activation[t].backward(dht)
            # np.clip(dtanh,-5,5,dtanh)
            
            self.dWy += np.dot(dy,self.H[t+1].T)
            
            self.dWx += np.dot(dtanh,xt)

            self.dbias += dtanh

            self.dWh += np.dot(dtanh,self.H[t].T)

            dht = np.dot(self.Wh.T, dtanh)
            # np.clip(dht,-5,5,dht)


class SGD_Optimizer:
    def __init__(self,learning_rate=1,decay=0,momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate=learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iteration = 0
    
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = (self.learning_rate)*(1.0/(1.+(self.decay*self.iteration)))

    def parameter_update(self,layer):
        np.clip(layer.dWx, -5, 5, out=layer.dWx)
        np.clip(layer.dWy, -5, 5, out=layer.dWy)
        np.clip(layer.dWh, -5, 5, out=layer.dWh)
        np.clip(layer.dbias, -5, 5, out=layer.dbias)
        
        if self.momentum:
            if not hasattr(layer,'momentum_Wx'):
                layer.momentum_Wx = np.zeros_like(layer.Wx)
                layer.momentum_Wh = np.zeros_like(layer.Wh)
                layer.momentum_Wy = np.zeros_like(layer.Wy)
                layer.momentum_bias = np.zeros_like(layer.bias)

            weight_updates_Wx = self.momentum * layer.momentum_Wx - \
                             self.current_learning_rate * layer.dWx
            layer.momentum_Wx = weight_updates_Wx

            weight_updates_Wy = self.momentum * layer.momentum_Wy - \
                             self.current_learning_rate * layer.dWy
            layer.momentum_Wy = weight_updates_Wy

            weight_updates_Wh = self.momentum * layer.momentum_Wh - \
                             self.current_learning_rate * layer.dWh
            layer.momentum_Wh = weight_updates_Wh

            bias_updates = self.momentum * layer.momentum_bias - \
                           self.current_learning_rate * layer.dbias
            layer.momentum_bias = bias_updates
            
    
        else:
            weight_updates_Wx = -self.current_learning_rate*layer.dWx
            weight_updates_Wy = -self.current_learning_rate*layer.dWy
            weight_updates_Wh = -self.current_learning_rate*layer.dWh
            bias_updates = -self.current_learning_rate*layer.dbias
            
        layer.Wx += weight_updates_Wx
        layer.Wy += weight_updates_Wy
        layer.Wh += weight_updates_Wh
        layer.bias += bias_updates
        
    def post_update(self):
            self.iteration += 1


class Adam_Optimizer:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iteration = 0

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))
    
    def parameter_update(self, layer):
        if not hasattr(layer, 'weight_cache_Wx'):
            layer.weight_cache_Wx = np.zeros_like(layer.Wx)
            layer.weight_cache_Wh = np.zeros_like(layer.Wh)
            layer.weight_cache_Wy = np.zeros_like(layer.Wy)
            layer.bias_cache = np.zeros_like(layer.bias)
            layer.momentum_Wx = np.zeros_like(layer.Wx)
            layer.momentum_Wh = np.zeros_like(layer.Wh)
            layer.momentum_Wy = np.zeros_like(layer.Wy)
            layer.momentum_bias = np.zeros_like(layer.bias)
        

        layer.weight_cache_Wx = layer.weight_cache_Wx * self.beta_2 + (1 - self.beta_2) * layer.dWx**2
        layer.weight_cache_Wh = layer.weight_cache_Wh * self.beta_2 + (1 - self.beta_2) * layer.dWh**2
        layer.weight_cache_Wy = layer.weight_cache_Wy * self.beta_2 + (1 - self.beta_2) * layer.dWy**2
        layer.bias_cache = layer.bias_cache * self.beta_2 + (1 - self.beta_2) * layer.dbias**2

        corrected_weight_cache_Wx = layer.weight_cache_Wx / (1 - self.beta_2 ** (self.iteration + 1))
        corrected_weight_cache_Wh = layer.weight_cache_Wh / (1 - self.beta_2 ** (self.iteration + 1))
        corrected_weight_cache_Wy = layer.weight_cache_Wy / (1 - self.beta_2 ** (self.iteration + 1))
        corrected_bias_cache = layer.bias_cache / (1 - self.beta_2 ** (self.iteration + 1))

        layer.momentum_Wx = layer.momentum_Wx * self.beta_1 + (1 - self.beta_1) * layer.dWx
        layer.momentum_Wh = layer.momentum_Wh * self.beta_1 + (1 - self.beta_1) * layer.dWh
        layer.momentum_Wy = layer.momentum_Wy * self.beta_1 + (1 - self.beta_1) * layer.dWy
        layer.momentum_bias = layer.momentum_bias * self.beta_1 + (1 - self.beta_1) * layer.dbias

        corrected_momentum_Wx = layer.momentum_Wx / (1 - self.beta_1 ** (self.iteration + 1))
        corrected_momentum_Wh = layer.momentum_Wh / (1 - self.beta_1 ** (self.iteration + 1))
        corrected_momentum_Wy = layer.momentum_Wy / (1 - self.beta_1 ** (self.iteration + 1))
        corrected_momentum_bias = layer.momentum_bias / (1 - self.beta_1 ** (self.iteration + 1))

       
        layer.Wx += -self.current_learning_rate * corrected_momentum_Wx / (np.sqrt(corrected_weight_cache_Wx) + self.epsilon)
        layer.Wh += -self.current_learning_rate * corrected_momentum_Wh / (np.sqrt(corrected_weight_cache_Wh) + self.epsilon)
        layer.Wy += -self.current_learning_rate * corrected_momentum_Wy / (np.sqrt(corrected_weight_cache_Wy) + self.epsilon)
        layer.bias += -self.current_learning_rate * corrected_momentum_bias / (np.sqrt(corrected_bias_cache) + self.epsilon)

    def post_update(self):
        self.iteration += 1

