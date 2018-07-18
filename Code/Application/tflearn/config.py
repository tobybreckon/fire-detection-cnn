from alexnet import *
from alexnetv1 import *
from alexnetv2 import *
from alexnetv3 import *
from alexnetv4 import *
from alexnetv5 import *
from alexnets import *
from googlenet import *
from googlenetv1 import *
from googlenetv2 import *
from googlenets import *
from vgg13 import *
from vgg16 import *
from customnet import *
from evaluate import *
from vgg13_retrain import *
from alexnetv3_retrain import *
from evaluatesp import *

#retrain_with_eval_alexnetv3 ("data100", ["/home/jrnf24/data/validation_data100.h5", "/home/jrnf24/data/validation_data_furg50.h5"])

model = construct_googlenet (224, 224)

def count_number_trainable_params():
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params

print (count_number_trainable_params())

#evaluate("/home/jrnf24/trained/G23/",["/home/jrnf24/data/validation_data100.h5", "/home/jrnf24/data/validation_data_furg50.h5"], model)
