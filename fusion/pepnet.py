import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Concatenate
from tensorflow.keras.models import Model

class DNNGate(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, hidden_activation='relu', dropout_rate=0.0):
        super(DNNGate, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        
        self.dense1 = Dense(hidden_dim)
        self.activation = Activation(hidden_activation)
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
        self.dense2 = Dense(output_dim)
        self.sigmoid = Activation('sigmoid')

    def call(self, feature_emb, gate_emb, mlp_out):
        gate_input = tf.concat([tf.stop_gradient(feature_emb), gate_emb], axis=1)
        g = self.dense1(gate_input)
        g = self.activation(g)
        print(tf.shape(g))
        if self.dropout:
            g = self.dropout(g)
        g = self.dense2(g)
        g = self.sigmoid(g) * 2
        print(tf.shape(g))
        print(tf.shape(mlp_out))
        h = mlp_out * g
        return h
@tf.function(jit_compile=True)
def test_ppnet_mlp(model,input,gate_emb,mlp_out):
    with tf.device('/GPU:0'):
        # component = DNNGate(
        #     input_dim=30*64,
        #     hidden_dim=64,
        #     output_dim=64,
        # )
        #input = tf.random.normal([1024, 27* 64])
        component=model
        output = component(input, gate_emb,mlp_out)
        
    return output

if __name__ == '__main__':
    input = tf.random.normal([1024, 27* 64])
    gate_emb=tf.random.normal([1024, 3*64])
    mlp_out=tf.random.normal([1024, 64])
    component = DNNGate(
            input_dim=30*64,
            hidden_dim=64,
            output_dim=64,
        )
    output = test_ppnet_mlp(component,input,gate_emb,mlp_out)
    print(output)
