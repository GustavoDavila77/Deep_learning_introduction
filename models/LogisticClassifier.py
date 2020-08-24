'''
Emotion classification with a Logistic classifier
@author: Omar U. Florez
'''

import tensorflow as tf
import ipdb


# Model:
#   Data transformation:
#       - [Batch size, seq_max_len] -> [Batch size, seq_max_len, vocab_size]
#       - [Batch size, seq_max_len, vocab_size] -> [Batch size, seq_max_len * vocab_size]
#   Neural Network architecture:
#       - output = [Batch size, seq_max_len * vocab_size] x [seq_max_len * vocab_size, num_classes]
#       - logits = Sigmoid(output)
#       - probabilities = Softmax(logits)
class LogisticClassifier:
    def __init__(self, seq_max_len, state_size, vocab_size, num_classes):
        # Maximum number of units in the input (e.g., words)
        self.seq_max_len = seq_max_len
        # Number of dimensions of the internal memory (for LSTM and MLP)
        self.state_size = state_size
        # Number of unique characters defining input data
        self.vocab_size = vocab_size
        # Number of output classes
        self.num_classes = num_classes

    def build_model(self):
        #significado de seq_max_len*vocab_size = es el tamaño de la matriz de entrada
        #el seq_max_len es la longitud max de palabras de los twits y representaria una dimensión de la matriz (column)
        #se multiplica por el tamaño del vocabulario porque esta representa otra dimensión (filas) donde va a ir 
        #las letras de cada palabra 

        #para entender el one_hot_encoding ver img en carpeta imagenes o buscar representación en internet 
        #Placeholder: represent input data
        #None because don´t import row size (twits number)
        #seq_max_len: longitud de los datos de entrada (palabras)
        self.x = tf.placeholder(shape=[None, self.seq_max_len], dtype=tf.int32)
        #one_hot_encoding: mean transform vocabulary in numbers, with the objective of the model can work with this
        # Convert word ids from the input into ortogonal vectors
        # Forming the input representation for the Logistic algorithm:
        #       - [Batch size, seq_max_len] -> [Batch size, seq_max_len, vocab_size]
        x_one_hot = tf.one_hot(self.x, self.vocab_size)
        x_one_hot = tf.cast(x_one_hot, tf.float32)


        # Forming the output representation for the Logistic algorithm using one-hot encoding:
        #       - [Batch size, 1] -> [Batch size, num_classes]
        self.y = tf.placeholder(shape=[None], dtype=tf.int32)
        self.y_onehot = tf.one_hot(self.y, self.num_classes, dtype=tf.float32)
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size') 

        # Define weights for output decoding 
        #each layer is trainable  
        #weights & biases is similar to y= mx + b where m is a mapping, x is data input and b is bias
        #Se necesita un valor cercano a cero para la inicialización de la variable, esto se logra con el random_normal
        weights = {
            'layer_0': tf.Variable(tf.random_normal([self.seq_max_len*self.vocab_size, self.num_classes])) 
        }
        biases = {
            'layer_0': tf.Variable(tf.random_normal([self.num_classes]))
        }

        #reshape is for model´s compatibility 
        x_input = tf.reshape(x_one_hot, [-1, self.seq_max_len*self.vocab_size])
        output = tf.matmul(x_input, weights['layer_0']) + biases['layer_0']
        self.logits = tf.sigmoid(output) #to add no linear linearity 
        self.probs = tf.nn.softmax(self.logits, axis=1) #to convert the result in probability 

        self.correct_preds = tf.equal(tf.argmax(self.probs, axis=1), tf.argmax(self.y_onehot, axis=1)) #compare best probability with real result
        self.precision = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32)) #mean of the correct prediction
        return

    def step_training(self, learning_rate=0.01):
        # softmax_cross_entropy_with_logits(): This function computes the probabilistic distance
        # between two distributions: the actual classes (y_onehot) and the predicted ones as log-probabilities
        # (self.logits). While the classes are mutually exclusive, their probabilities need not be.
        # All that is required is that each row of labels is a valid probability distribution.
        # If they are not, the computation of the gradient will be incorrect.
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_onehot,
                                                                      logits=self.logits))
        #adamOptimizer is necesary for fix silla de montar/saddle point problem 
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return loss, optimizer






