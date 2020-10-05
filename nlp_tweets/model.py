from tensorflow.keras.layers import Dense, Dropout, Activation, Input, GRU, TimeDistributed, \
    Embedding, BatchNormalization, Flatten
from tensorflow.keras.models import Model
import tensorflow


def get_model(vocabulary_size: int, embeding_dim: int, max_sequence_length: int,
              _rnn_nb: [int], _fc_nb: [int], dropout_rate: float) -> tensorflow.keras.models.Model:
    sequence_1_input = Input(shape=(max_sequence_length,), dtype='int32')

    seq = Embedding(vocabulary_size, embeding_dim, input_length=max_sequence_length, trainable=False)(sequence_1_input)
    for _r in _rnn_nb:
        seq = GRU(_r, activation='tanh', dropout=dropout_rate,
                  recurrent_dropout=dropout_rate, return_sequences=True)(seq)

    for _f in _fc_nb:
        seq = TimeDistributed(Dense(_f))(seq)
        seq = Dropout(dropout_rate)(seq)
        seq = TimeDistributed(Dense(_f))(seq)

    seq = Flatten()(seq)
    seq = Dense(10)(seq)
    seq = Activation('relu')(seq)
    seq = Dropout(dropout_rate)(seq)
    seq = BatchNormalization()(seq)
    seq = Dense(1)(seq)
    out = Activation('sigmoid', name='strong_out')(seq)

    model = Model(inputs=sequence_1_input, outputs=out)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
