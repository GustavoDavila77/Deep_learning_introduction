'''
Emotion classification with Multi Layer Perceptron (MLP)
@author: Omar U. Florez
'''

import tensorflow as tf
import ipdb


# Model:
#   Data transformation:
#       - [Batch size, seq_max_len] -> [Batch size, seq_max_len, vocab_size]
#       - [Batch size, seq_max_len, vocab_size] -> [Batch size, seq_max_len * vocab_size]
#   Neural Network architecture:
#       - hidden = [Batch size, seq_max_len * vocab_size] x [seq_max_len * vocab_size, state_size]
#       - hidden = Sigmoid(hidden)
#       - hidden = [Batch size, state_size] x [state_size, state_size]
#       - hidden = Sigmoid(hidden)
#       - logits = [state_size, state_size] x [state_size, num_classes]
#       - probabilities = Softmax(logits)
class MLP:
    def __init__(self, seq_max_len, state_size, vocab_size, num_classes):
        # Maximum number of units in the input (e.g., words)
        self.seq_max_len = seq_max_len
        # Number of dimensions of the hidden layer
        self.state_size = state_size
        # Number of unique words in the input domain
        self.vocab_size = vocab_size
        # Number of output classes
        self.num_classes = num_classes

    def build_model(self):
        self.x = tf.placeholder(shape=[None, self.seq_max_len], dtype=tf.int32)
        # Convert word ids from the input into orthogonal vectors via one-hot encoding
        x_one_hot = tf.one_hot(self.x, self.vocab_size)
        x_one_hot = tf.cast(x_one_hot, tf.float32)

        # Convert class ids into orthogonal vectors via one-hot encoding
        self.y = tf.placeholder(shape=[None], dtype=tf.int32)
        self.y_onehot = tf.one_hot(self.y, self.num_classes, dtype=tf.float32)

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        # Define weights
        #seq_max_len: longitud de los datos de entrada
        #la capa escondida esta representada por la dimensión state_size 
        weights = {
            'layer_0': tf.Variable(tf.random_normal([self.seq_max_len*self.vocab_size, self.state_size])),
            'layer_1': tf.Variable(tf.random_normal([self.state_size, self.state_size])),
            'layer_2': tf.Variable(tf.random_normal([self.state_size, self.num_classes])) #num_classes = numero de neurnoas en la última capa 
        }
        # Define bias weights
        #bias es componente que dirige la predicción. es un forma de dirigir la tendencia
        #el tamaño de la capa 0,1 seran proporcionales al tamaño del state_size(capa oculta)
        biases = {
            'layer_0': tf.Variable(tf.random_normal([self.state_size])),
            'layer_1': tf.Variable(tf.random_normal([self.state_size])),
            'layer_2': tf.Variable(tf.random_normal([self.num_classes]))
        }

        #se hace el reshape para poder hacer la multiplicación de matrices
        #se lleva el x_one_hot a una forma más sencilla -1: no importa el número de filas
        #camponente de entrada
        x_input = tf.reshape(x_one_hot, [-1, self.seq_max_len*self.vocab_size])
        hidden = tf.matmul(x_input, weights['layer_0']) + biases['layer_0']
        hidden = tf.nn.sigmoid(hidden) #se añade no linealidad - salida de la capa de entrada
        
        #componente escondido
        hidden = tf.matmul(hidden, weights['layer_1']) + biases['layer_1']
        hidden = tf.nn.sigmoid(hidden)
        
        #componente de salida  
        self.logits = tf.matmul(hidden, weights['layer_2']) + biases['layer_2']
        self.probs = tf.nn.softmax(self.logits, axis=1) #convertir a un conjunto de probabilidades 

        #observar las predicciones correctas y mirar precisión
        self.correct_preds = tf.equal(tf.argmax(self.probs, axis=1), tf.argmax(self.y_onehot, axis=1)) #escogemos las mejores probabilidades 
        self.precision = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))
        return

    #optimizar el modelo 
    def step_training(self, learning_rate=0.01):
        # softmax_cross_entropy_with_logits(): This function computes the probabilistic distance
        # between two distributions: the actual classes (y_onehot) and the predicted ones as log-probabilities
        # (self.logits). While the classes are mutually exclusive, their probabilities need not be.
        # All that is required is that each row of labels is a valid probability distribution.
        # If they are not, the computation of the gradient will be incorrect.
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_onehot,
                                                                      logits=self.logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return loss, optimizer #minimiza la función de costo






