from keras.layers import Input, Dense, Activation, Lambda, Conv1D, Flatten, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras import backend as K


class CNN_Model:
    def __init__(self, config, hidden_size=[128, 64], kernal_size=7, kernal_num=32, denseactivation='relu'):
        # here the hyperparameters are set to the reported best ones in our paper
        # feel free to change them for your convenience
        self.output_dim = config['output_dim']
        self.feature_len = config['feature_len']
        self.kernal_size = kernal_size
        self.hidden_size = hidden_size
        self.kernal_num = kernal_num
        self.denseactivation = denseactivation

        self.model = None
        self.build()

    def build(self):
        input_feature = Input(shape=(self.feature_len,))

        input_feature1 = Lambda(lambda x: K.expand_dims(x, axis=2))(input_feature)

        cnn_feature = Conv1D(self.kernal_num, self.kernal_size, activation='relu', input_shape=[self.feature_len, 1])(input_feature1)  # [17, 32]
        '''
            adding pooling layer may or may not benefit the model performance, 
            depending on specific tasks and datasets
            if needed, feel free to uncomment the following codes
        '''
        # cnn_feature = MaxPooling1D(2, 1, 'valid')(cnn_feature)
        # cnn_feature = AveragePooling1D(2, 1, 'valid')(cnn_feature)        

        cnn_feature_1 = Flatten()(cnn_feature)
        for i in range(len(self.hidden_size)):
            cnn_feature_1 = Dense(self.hidden_size[i], activation=self.denseactivation)(cnn_feature_1)
        output = Dense(self.output_dim, activation='sigmoid')(cnn_feature_1)

        # Model
        self.model = Model(inputs=input_feature, outputs=output)
