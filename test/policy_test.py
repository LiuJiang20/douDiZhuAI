from models.neural_nets import SimpleCov
NEURAL_NET_CLASS = SimpleCov.__name__
EXTENSION = '.pt'
LANDLORD_MODEL_PATH = 'models/landlord_model' + NEURAL_NET_CLASS + EXTENSION
landlord_model = SimpleCov().cuda()

landlord_model.forward()