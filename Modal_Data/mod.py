import keras.backend as K
import numpy as np
from keras.layers import *
from keras.models import *
import keras.backend.tensorflow_backend as KTF
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from LSTM_SA import SA_LSTM
# from GRU_SA import SA_GRU
from keras.layers import StackedRNNCells
from keras import regularizers, constraints, initializers, activations
import scipy.io as sio
from keras import optimizers
import matplotlib.pyplot as plt
from keras.utils import plot_model
# from IndRNN import IndRNN
from keras.initializers import RandomUniform
from keras.constraints import MaxNorm
from sklearn.metrics.pairwise import cosine_similarity

T = 5000
limit = 2 ** (1 / T)

# 设定为自增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)
slim = tf.contrib.slim


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def create_dataset(data, n_predictions, n_next):
    dim = 1  # dim=3
    train_X, train_Y = [], []
    for i in range(data.shape[0] - n_predictions - n_next - 1):  # 8356-历史数据-预测天数-1=8318
        a = data[i:(i + n_predictions), :, :]  # 0-30,3/1-31,3
        train_X.append(a)
        tempb = data[(i + n_predictions):(i + n_predictions + n_next), :, :]  # 30-37,3
        train_Y.append(tempb)
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')

    return train_X, train_Y


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    if not input_dim:  # 判空，if not x：x空为真
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]  # 一种高端的赋值操作
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:  # if x is not None：x不空为真
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))  # 创建同样维度的全1向量
        dropout_matrix = K.dropout(ones, dropout)  # dropout是丢掉的比率
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)  # 重复dropout_matrix操作 timesteps 次
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)  # 在训练阶段返回 x*expended_dropout_matrix更新x，
        # 非训练阶段，x不更新
    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))  # 将timestamp和batchsize合二为1
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


class sequence_CNN(Layer):

    def __init__(self, units, kernal):
        super(sequence_CNN, self).__init__()
        self.units = units
        self.kernal = kernal

    def call(self, inputs):
        timesteps = int(inputs.shape.as_list()[1])
        result = Conv2D(self.units, self.kernal, padding='same')(inputs[:, 0])
        result = tf.expand_dims(result, 1)
        for i in range(1, timesteps):
            conv_1 = Conv2D(self.units, self.kernal, padding='same')(inputs[:, i])
            conv_1 = tf.expand_dims(conv_1, 1)
            result = tf.concat([result, conv_1], 1)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.units


class sequence_Maxpool(Layer):

    def __init__(self):
        super(sequence_Maxpool, self).__init__()

    def call(self, inputs):
        timesteps = int(inputs.shape[1])
        result = AveragePooling2D(pool_size=(inputs.shape[2], inputs.shape[3]), strides=(1, 1), padding='valid')(
            inputs[:, 0])
        result = tf.expand_dims(result, 1)
        for i in range(1, timesteps):
            conv_1 = AveragePooling2D(pool_size=(inputs.shape[2], inputs.shape[3]), strides=(1, 1), padding='valid')(
                inputs[:, i])
            conv_1 = tf.expand_dims(conv_1, 1)
            result = tf.concat([result, conv_1], 1)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], 1, 1, input_shape[4]


class sequence_Avgpool(Layer):

    def __init__(self):
        super(sequence_Avgpool, self).__init__()

    def call(self, inputs):
        timesteps = int(inputs.shape[1])
        result = MaxPooling2D(pool_size=(inputs.shape[2], inputs.shape[3]), strides=(1, 1), padding='valid')(
            inputs[:, 0])
        result = tf.expand_dims(result, 1)
        for i in range(1, timesteps):
            conv_1 = MaxPooling2D(pool_size=(inputs.shape[2], inputs.shape[3]), strides=(1, 1), padding='valid')(
                inputs[:, i])
            conv_1 = tf.expand_dims(conv_1, 1)
            result = tf.concat([result, conv_1], 1)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], 1, 1, input_shape[4]

def cbam_module(inputs):
    # channel attention
    avgpool_channel = sequence_Avgpool()(inputs)
    maxpool_channel = sequence_Maxpool()(inputs)
    avg_pool = Dense(16, activation='relu')(avgpool_channel)
    max_pool = Dense(16, activation='relu')(maxpool_channel)
    avg_pool = Dense(32, activation='relu')(avg_pool)
    max_pool = Dense(32, activation='relu')(max_pool)
    max_pool = Dense(32, activation='relu')(max_pool)
    fc_2 = tf.nn.sigmoid(avg_pool + max_pool)
    channel_attention_module = tf.multiply(inputs, fc_2)

    # spatial attention
    inputs1 = Dense(32, activation='relu')(inputs)
    inputs1 = Dense(64, activation='relu')(inputs1)
    savg_pool = tf.reduce_mean(inputs1, axis=4, keep_dims=True)  # keep_dims:是否降维
    smax_pool = tf.reduce_max(inputs, axis=4, keep_dims=True)
    channel_w_pool = tf.concat([savg_pool, smax_pool], axis=4)

    spatial_attention = sequence_CNN(units=1, kernal=1)(channel_w_pool)  # 降维
    spatial_attention = tf.nn.sigmoid(spatial_attention)
    spatial_attention_module = tf.multiply(inputs, spatial_attention)

    # return spatial_attention_module
    return channel_attention_module


def cbam_module_ca(inputs):

    # channel attention
    avgpool_channel = AveragePooling2D(pool_size=(inputs.shape[1], inputs.shape[2]), strides=(1, 1), padding='valid')(inputs)
    maxpool_channel = MaxPooling2D(pool_size=(inputs.shape[1], inputs.shape[2]), strides=(1, 1), padding='valid')(inputs)
    avg_pool = Dense(16, activation='relu')(avgpool_channel)
    max_pool = Dense(16, activation='relu')(maxpool_channel)
    avg_pool = Dense(32)(avg_pool)
    max_pool = Dense(32)(max_pool)
    fc_2 = tf.nn.sigmoid(avg_pool + max_pool)
    channel_attention_module = tf.multiply(inputs, fc_2)

    return channel_attention_module


def cbam_module_sa(inputs):
    # spatial attention
    savg_pool = tf.reduce_mean(inputs, axis=3)
    smax_pool = tf.reduce_max(inputs, axis=3)
    savg_pool = Lambda(reshape_tensor, arguments={'shape': (-1, inputs.shape[1], inputs.shape[2], 1)})(savg_pool)
    smax_pool = Lambda(reshape_tensor, arguments={'shape': (-1, inputs.shape[1], inputs.shape[2], 1)})(smax_pool)
    channel_w_pool = tf.concat([savg_pool, smax_pool], axis=3)

    spatial_attention = Conv2D(filters=1, kernel_size=(7, 7), padding='same')(channel_w_pool)
    spatial_attention = tf.nn.sigmoid(spatial_attention)
    spatial_attention_module = tf.multiply(inputs, spatial_attention)

    return spatial_attention_module


def connection_Attention(inputs):
    avgpool_channel = sequence_Avgpool()(inputs)
    maxpool_channel = sequence_Maxpool()(inputs)
    avg_pool = Dense(16, activation='relu')(avgpool_channel)
    max_pool = Dense(16, activation='relu')(maxpool_channel)
    avg_pool = Dense(4, activation='relu')(avg_pool)
    max_pool = Dense(4, activation='relu')(max_pool)
    max_pool = Dense (4, activtion='relu')(max_pool)
    fc_2 = tf.nn.sigmoid(avg_pool + max_pool)
    print("fc_2:", fc_2.shape)
    channel_attention_module = tf.multiply(inputs, fc_2)
    return channel_attention_module


class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel', shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        print("x:", x.shape)
        W_q = K.dot(x, self.kernel[0])
        W_k = K.dot(x, self.kernel[1])
        W_v = K.dot(x, self.kernel[2])
        print("WQ.shape", W_q.shape)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(W_k, [0, 2, 1]).shape)
        QK = K.batch_dot(W_q, K.permute_dimensions(W_k, [0, 2, 1]))
        QK = QK / (64 ** 0.5)
        QK = K.softmax(QK)
        print("QK.shape", QK.shape)
        V = K.batch_dot(QK, W_v)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


class SAM(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SAM, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(5, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        self.rec_kernel = self.add_weight(name='rec_kernel',
                                          shape=(4, input_shape[2], self.output_dim),
                                          initializer='uniform',
                                          trainable=True)

        super(SAM, self).build(input_shape)  # 一定要在最后调用它

    def call(self, states):
        print("states.shape:", states.shape)
        h_t = states[0]
        m_t_1 = states[1]
        # 特征聚合
        Q_h = K.dot(h_t, self.kernel[0])
        K_h = K.dot(h_t, self.kernel[1])
        V_h = K.dot(h_t, self.kernel[2])
        K_m = K.dot(m_t_1, self.kernel[3])
        V_m = K.dot(m_t_1, self.kernel[4])
        print("Q_h.shape", Q_h.shape)
        print("K_h.shape", K_h.shape)
        print("Q.permute_dimensions(K_h, [0, 2, 1]).shape", K.permute_dimensions(K_h, [0, 2, 1]).shape)
        A_h = K.batch_dot(Q_h, K.permute_dimensions(K_h, [0, 2, 1]))
        A_h = A_h / (64 ** 0.5)
        A_h = K.softmax(A_h)
        Z_h = K.batch_dot(A_h, V_h)

        A_m = K.batch_dot(K_m, K.permute_dimensions(K_h, [0, 2, 1]))
        A_m = A_m / (64 ** 0.5)
        A_m = K.softmax(A_m)
        Z_m = K.batch_dot(A_m, V_m)

        Z = K.dot((tf.concat([Z_h, Z_m], axis=1)), self.rec_kernel[0])
        print("Z.shape:", Z.shape)
        Z = tf.concat([Z, h_t], axis=1)

        # 记忆更新
        m_o = K.dot(Z, self.rec_kernel[1])
        m_g = K.dot(Z, self.rec_kernel[2])
        m_i = K.sigmoid(K.dot(Z, self.rec_kernel[3]))
        m_t = K.tanh(m_g) * m_i + (1 - m_i) * m_t_1
        h = m_t * K.sigmoid(m_o)

        return h, [h, m_t]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim, 2)


SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # inputs = Activation("tanh")(inputs)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)  # 转置函数
    a = Dense(TIMESTEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


class CosineLayer():
    # x1,x2:(batch_size, dim)
    # cosine = CosineLayer()
    # similarity = cosine(x1, x2)
    def __call__(self, x1, x2):
        def _cosine(x):
            dot1 = K.batch_dot(x[0], x[1], axes=1)
            dot2 = K.batch_dot(x[0], x[0], axes=1)
            dot3 = K.batch_dot(x[1], x[1], axes=1)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            return dot1 / max_

        output_shape = (1,)
        value = Lambda(_cosine, output_shape=output_shape)([x1, x2])
        return value


class attention_nearly_Layer(Layer):

    def __init__(self, units):  # W_regularizer=None, W_constraint=None, **kwargs
        super(attention_nearly_Layer, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

        # self.init = initializers.get('glorot_uniform')
        # self.W_regularizer = regularizers.get(W_regularizer)
        # self.W_constraint = constraints.get(W_constraint)
        # self.features_dim = 0

    # def build(self, input_shape):
    # self.features_dim = input_shape[-1]
    # self.W = self.add_weight((self.features_dim, ),  initializer=self.init, name='{}_W'.format(self.name),
    # regularizer=self.W_regularizer, constraint=self.W_constraint)

    # self.built = True
    '''
            query = connect_input[:, 7, :]
            inputs = connect_input[:, :7, :]
            x = connect_input[:, 0, :]
            connect = CosineLayer()(x, query)
            #connect = K.dot(x, K.reshape(self.W, (self.features_dim, 1)))
            #connect = cosine_similarity(x, query)
            connect = tf.expand_dims(connect, axis=1)

            for i in range(1, TIMESTEPS):
                x1 = connect_input[:, i, :]
                similarity = CosineLayer()(x1, query)
                #similarity = K.dot(x1, K.reshape(self.W, (self.features_dim, 1)))
                similarity = tf.expand_dims(similarity, axis=1)
                connect = tf.concat([connect, similarity], axis=1)

            a = Permute((2, 1))(connect)  # 转置函数
            a = Dense(TIMESTEPS, activation='softmax')(a)
            a_probs = Permute((2, 1), name='attention_vec')(a)
            output_attention_mul = Multiply()([inputs, a_probs])

            return output_attention_mul
    '''

    def call(self, query, valus):
        # hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(K.tanh(self.W1(valus) + self.W2(query)))
        attention_weights = K.softmax(score, axis=1)

        context_vector = attention_weights * valus
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Nearly_Attention(Layer):
    def __int__(self, use_scale=False, **kwargs):
        super(Nearly_Attention, self).__int__(**kwargs)
        self.use_scale = use_scale

    def build(self, input_shape):
        if self.use_scale:
            self.scale = self.add_weight(name='scale', shape=(), initializer=tf.compat.v1.ones_initializer(),
                                         dtype=self.dtype, trainable=True)
        else:
            self.scale = None

        self.built = True

    def calculate_score(self, query, key):
        scores = tf.matmul(query, key, transpose_b=True)
        if self.scale is not None:
            scores *= self.scale
        return scores

    def get_config(self):
        config = {'use_scale': self.use_scale}
        base_config = super(Nearly_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, hidden_states):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(2048, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

class Mid_Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, hidden_states):

        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec1')(hidden_states)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(OUTPUTDIM, hidden_size), name='last_hidden_state1')(hidden_states)
        h_t = Lambda(reshape_tensor, arguments={'shape': (-1, OUTPUTDIM, hidden_size)})(h_t)
        print("h_t:", h_t.shape)
        score = dot([score_first_part, h_t], [2, 2], name='attention_score1')
        print("score:", score.shape)
        attention_weights = Activation('softmax', name='attention_weight1')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = multiply([attention_weights, hidden_states])
        print("context_vector:", context_vector.shape)
        pre_activation = add([context_vector, h_t], name='attention_output1')
        attention_vector = Dense(2048, use_bias=False, activation='tanh', name='attention_vector1')(pre_activation)
        return attention_vector

        return context_vector



class Bahdanau_Attention(Layer):

    def __init__(self, units, **kwargs):
        super(Bahdanau_Attention, self).__init__(**kwargs)
        self.units = units

    def __call__(self, hidden_states):
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(self.units, use_bias=False, name='attention_score_vec1')(hidden_states)
        print("first:", score_first_part.shape)
        query = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score_second_part = Dense(self.units, use_bias=False, name='attention_score_vec2')(hidden_with_time_axis)
        score_third_part = tf.nn.tanh(score_first_part + score_second_part)
        score = Dense(1, use_bias=False, name='attention_score_vec3')(score_third_part)
        # score = tf.matmul(hidden_states, hidden_with_time_axis, transpose_b=True)
        print("score:", score.shape)
        attention_weights = Activation('softmax', name='attention_weight')(score)
        context_vector = hidden_states * attention_weights
        context_vector = GlobalAveragePooling1D()(context_vector)
        # context_vector = tf.reduce_sum(context_vector, axis=1) #等同于全局池化，效果不好
        pre_activation = concatenate([context_vector, query], name='attention_output')
        # print("pre_activation:", pre_activation.shape)
        attention_vector = Dense(2048, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)

        return attention_vector


def model_attention_applied_after_lstm():
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(look_back_x * 25, 1))
    lstm_units = 6
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    # print("lstm_out.shape:", lstm_out.shape)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(25, activation='tanh')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def trainModel(output_dim=25):

    '''
    inputs = Input(shape=(local, local, TIMESTEPS))

    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', name='conv1_1')(inputs)
    conv1_1 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv1_1)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', name='conv1_2')(conv1_1)
    conv1_2 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv1_2)
    conv1_2 = BatchNormalization()(conv1_2)
    #print("conv:", conv1_2.shape)

    #attention_mulca = Lambda(cbam_module_ca, name='cbam_module_ca')(conv1_2)
    #attention_mulsa = Lambda(cbam_module_sa, name='cbam_module_sa')(conv1_2)
    #print("attention_mulca:", attention_mulca.shape)
    #print("attention_mulsa:", attention_mulsa.shape)

    #attention_mul = concatenate([attention_mulca, attention_mulsa])
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv2_1')(conv1_2)
    conv2_1 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,shared_axes=None)(conv2_1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv2_2')(conv2_1)
    conv2_2 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,shared_axes=None)(conv2_2)

    spatial_module = Lambda(reshape_tensor, arguments={'shape': (-1, local*local*64)})(conv2_2)

    # 获取空间特征部分
    inputs = Input(shape=(TIMESTEPS, local, local, 1))
    # 一层卷积层，包含了32个卷积核，大小为3*3

    conv1_1 = sequence_CNN(32, kernal=3)(inputs)
    conv1_1 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv1_1)

    conv1_2 = sequence_CNN(32, kernal=3)(conv1_1)
    conv1_2 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv1_2)

    conv1_2 = BatchNormalization()(conv1_2)

    #attention_mulca = Lambda(cbam_module_ca, name='cbam_module1')(conv1_2)

    conv2_1 = sequence_CNN(64, kernal=3)(conv1_2)
    conv2_1 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv2_1)

    # 添加一个卷积层，包含64个卷积和，每个卷积和仍为3*3
    conv2_2 = sequence_CNN(1, kernal=3)(conv2_1)
    conv2_2 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv2_2)
    spatial_fea = Lambda(reshape_tensor, arguments={'shape': (-1, TIMESTEPS, output_dim)})(conv2_2)
    spatial_fea = Lambda(lambda x: x[:, -1, :], output_shape=(output_dim,), name='last_hidden_spastate')(spatial_fea)

'''

    inputs = Input(shape=(TIMESTEPS, local, local, 1))

    conv1_1 = Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name='conv1_1')(inputs) #默认设置 chnnels在最后一维
    conv1_1 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv1_1)
    conv1_2 = Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name='conv1_2')(conv1_1)  # 默认设置 chnnels在最后一维
    conv1_2 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv1_2)
    print("conv3D:", conv1_2.shape)
    conv1_2 = BatchNormalization()(conv1_2)#shape:(?,timesteps,5,5,32)


    conv2_1 = sequence_CNN(32, kernal=3)(inputs)
    conv2_1 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv2_1)

    conv2_2 = sequence_CNN(32, kernal=3)(conv2_1)
    conv2_2 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv2_2)

    conv2_2 = BatchNormalization()(conv2_2) #shape:(?,timesteps,5,5,32)
    print("conv2_2:", conv2_2.shape)
    attention_mulca = Lambda(cbam_module, name='cbam_module1')(conv2_2)


    c_addc = concatenate([conv1_2, attention_mulca])
    conv3_1 = sequence_CNN(1, kernal=3)(c_addc)
    conv3_1 = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv3_1)
    spatial_fea = Lambda(reshape_tensor, arguments={'shape': (-1, TIMESTEPS, output_dim)})(conv3_1)
    spatial_fea = Lambda(lambda x: x[:, -1, :], output_shape=(output_dim,), name='last_hidden_spastate')(spatial_fea)

    inputs1 = Input(shape=(TIMESTEPS, output_dim))
    simp_rnn = GRU(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, implementation=2, name='gru1')(inputs1)
    simp_rnn = Attention()(simp_rnn)
    dense = concatenate([spatial_fea, simp_rnn])
    dense = Dense(256, activation='linear')(dense)

    output = Dense(output_dim, activation='tanh')(dense)  # linear;tanh;sigmoid
    model = Model(inputs=[inputs, inputs1], outputs=output)
    print(model.summary())

    '''
    # 模型网络结构图
    plot_model(model, to_file='model1.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    plt.figure(figsize=(20, 20))
    img = plt.imread('model1.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
'''
    return model


def reshape_tensor(x, shape):
    return K.reshape(x, shape)


TIMESTEPS = 7
OUTPUTDIM = 1
batch_size = 300
NUM_HIDDENUNITS = 6
NUM_EPOCH = 100
RECURRENT_MAX = pow(2, 1 / TIMESTEPS)

# load the dataset
v = np.load("v.npy")
n, h, w = v.shape


look_back_x = TIMESTEPS
look_back_y = OUTPUTDIM
SINGLE_ATTENTION_VECTOR = False
local = 5
rows = int(h / local)
cols = int(w / local)

rMSE_total1 = 0
acc_total1 = 0
MAE_total1 = 0
r_total = 0
num1 = 0
number = 0

'''
x1 = np.reshape(v, (585200, 1))

#dataset preprocess
dataframe1 = pd.DataFrame(x1)
dataset1 = dataframe1.values
dataset1 = dataset1.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(x1)

dataset = np.reshape(dataset1, (1463, 20, 20))
'''


for i in range(0, rows):
    for j in range(0, cols):
        number += 1
        print(str(number) + "/" + str(rows * cols) + " Station—— （" + str(i + 1) + "，" + str(j + 1) + "）：\n")
        data_set = v[:, i:i + local, j:j + local]
        data_Set = np.reshape(data_set, (36575, 1))
        # dataset preprocess
        dataframe1 = pd.DataFrame(data_Set)
        dataset1 = dataframe1.values
        dataset1 = dataset1.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset1 = scaler.fit_transform(data_Set)
        dataset = np.reshape(dataset1, (v.shape[0], local, local))


        # split into train and test sets
        train, test = dataset[0:1096, :, :], dataset[1096:1463, :, :]
        # 制作数据集
        trainX, trainY = create_dataset(train, look_back_x, look_back_y)
        length = len(trainX) // batch_size * batch_size
        trainX = trainX[0:length]
        trainY = trainY[0:length]
        print("trainX.shape:", trainX.shape)
        print("trainY.shape:", trainY.shape)
        testX, testY = create_dataset(test, look_back_x, look_back_y)
        length = len(testX) // batch_size * batch_size
        testX = testX[0:length]
        testY = testY[0:length]

        # create and fit the  network\\\\reshape input to be [samples, time steps, features]
        # 输入空间部分的数据集格式
        trainX_S = np.reshape(trainX, (trainX.shape[0], TIMESTEPS, local, local, 1))
        print("trainX_S.shape:", trainX_S.shape)
        testX_S = np.reshape(testX, (testX.shape[0], TIMESTEPS, local, local, 1))

        # 输入时间部分数据集的格式
        trainX_T = np.reshape(trainX, (trainX.shape[0], TIMESTEPS, local * local))
        print("trainX_T.shape:", trainX_T.shape)
        testX_T = np.reshape(testX, (testX.shape[0], TIMESTEPS, local * local))

        trainY = np.reshape(trainY, (trainY.shape[0] * OUTPUTDIM, local * local))
        testY = np.reshape(testY, (testY.shape[0] * OUTPUTDIM, local * local))
        trainY_T = np.reshape(trainY, (trainY.shape[0], OUTPUTDIM, local * local))
        testY_T = np.reshape(testY, (testY.shape[0], OUTPUTDIM, local * local))

        m = trainModel()
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        #sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        #sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
        #sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
        #rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) #其余使用默认参数，lr可调节
        #adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        #nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        # m.compile(loss=tf.keras.losses.Huber(), optimizer='adam')  # loss值Huber
        m.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
        hist = m.fit([trainX_S, trainX_T], trainY, nb_epoch=NUM_EPOCH, batch_size=batch_size, verbose=2, shuffle=False,
                     validation_split=0.12048)

        # 对测试集进行预测
        trainPredict = m.predict([trainX_S, trainX_T], batch_size=batch_size)
        testPredict = m.predict([testX_S, testX_T], batch_size=batch_size)

        testY_ = testY
        testPredict_ = testPredict
        print("预测结果如下：")
        acc_u = 1 - np.abs(np.mean(np.abs(testPredict_[:, :] - testY_[:, :]) / np.abs(testY_[:, :])))
        print("ACC:", acc_u)

        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform(trainY)
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform(testY)

        '''
        # plot
        filename = str(i) + "_" + str(j)
        fig, ax = plt.subplots(1)
        test_values = testY[:, 0].reshape(-1, 1).flatten()
        plot_test, = ax.plot(test_values)
        predicted_values = testPredict[:, 0].reshape(-1, 1).flatten()
        plot_predicted, = ax.plot(predicted_values)
        plt.title('U Predictions')
        plt.legend([plot_predicted, plot_test], ['predicted', 'true value'])
        plt.savefig(filename + '_predict')
        plt.show()

'''
        # scores
        MSE1 = mean_squared_error(testY[:, :], testPredict[:, :])  # 均方误差
        rMSE1 = math.sqrt(MSE1)
        print("rMSE:%.6f:", rMSE1)
        MAE1 = mean_absolute_error(testY[:, :], testPredict[:, :])  # 平均绝对误差
        print("MAE:%.6f:", MAE1)
        R1 = 1 - mean_squared_error(testY[:, :], testPredict[:, :])/ np.var(testY[:, :]) #R-square
        print("R-square:%.6f:", R1)

        # sum
        acc_total1 = acc_total1 + acc_u
        rMSE_total1 = rMSE_total1 + rMSE1
        MAE_total1 = MAE_total1 + MAE1
        r_total = r_total + R1

print('有效区域块:', number)

rMSE_ave1 = rMSE_total1 / (rows * cols)
acc_ave1 = acc_total1 / (rows * cols)
MAE_ave1 = MAE_total1 / (rows * cols)
r_avg = r_total / (rows * cols)
print("U-----------RMSE_ave:", rMSE_ave1, "MAE_ave:", MAE_ave1, "ACC_ave:", acc_ave1, "r_avg:", r_avg)


