from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model

def _override_last_dense_layer(model):
    input_layer = model.layers[0].input
    last_layer = model.layers[-2].output
    last_layer = Dense(1, name='class_rating_dense', activation='linear')(last_layer)
    return Model(inputs=[input_layer], outputs=[last_layer])

def compile_model_decorator(fn):
    def wrapper(*args, **kwargs):
        model = fn(*args, **kwargs)
        model.compile(optimizer='adam',
            loss='mse', metrics=['mape'])
        return model
    return wrapper

@compile_model_decorator
def create_regression_rating_model_from_pretrained_model(pretrained_model):
    return _override_last_dense_layer(pretrained_model)