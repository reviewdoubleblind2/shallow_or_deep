import warnings

from keras.backend import clear_session
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import os
from keras.utils import np_utils
import joblib
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.layers import GlobalMaxPooling1D, Conv1D, Embedding
import pickle
from keras.models import load_model

import optuna

#number of epochs needed for training
EPOCHS= 100
#Length of the vocabulary per each approach, dataset and representation
# eg for binary, Russell, b0 is 150
VOCAB= 150
#Approach it can be binary 
approach = 'binary'
# Dataset Russell (R-DS), Juliet (J-DS) and OUR (GH-DS),
dataset= 'Russell'
# Representation b0 (R0), b1 (R1), b1_int (R2), b1_iden (R3), b1_int_iden (R4)
rep= 'b0'
# The path where the folder of the data (X and Y) is located
base_path= ''
#Model CNN
NN= 'CNN'

def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    X_train = np.load(os.path.join(base_path,'Sequence_tokens',f'{approach}_{dataset}_train_{rep}.npy'))
    X_test = np.load(os.path.join(base_path,'Sequence_tokens',f'{approach}_{dataset}_val_{rep}.npy'))

    if approach == 'binary':
        df_Y_train = pd.read_csv(os.path.join(base_path,'Data_train_test',f'{approach}_Ytrain_{dataset}_{rep}.csv'))
        df_Y_train['VULN_N'] = df_Y_train['VULN_N'].apply(lambda x: 1 if x=='VULN' else 0)
        Y_train = df_Y_train['VULN_N']
        df_Y_test = pd.read_csv(os.path.join(base_path,'Data_train_test',f'{approach}_Yval_{dataset}_{rep}.csv'))
        df_Y_test['VULN_N'] = df_Y_test['VULN_N'].apply(lambda x: 1 if x=='VULN' else 0)
        Y_test = df_Y_test['VULN_N']

    vocab_size = VOCAB
    model = Sequential() # first model
    model.add(Embedding(input_dim = vocab_size, output_dim= trial.suggest_categorical("out_put", [8,16,32,64,128]), input_length=X_train.shape[1]))
    number_layers = trial.suggest_categorical("num_layers", [1,3])

    if number_layers == 1:
        s_f_1 = trial.suggest_categorical("filter_size0", [8,16,32,64,128])
    else:
        s_f_1 = trial.suggest_categorical("filter_size1", [32,64,128])

    s_k_1 = trial.suggest_categorical("kernel_size1", [1,2,3])
    st_1 = trial.suggest_categorical("strides1", [1,3,5,7])
    i_k_1= trial.suggest_categorical("kernel_init1",['Orthogonal','lecun_uniform','he_normal'])
    a_c_1= trial.suggest_categorical("activation_conv1",['relu','tanh','sigmoid'])
    
    model.add(Conv1D(filters= s_f_1, kernel_size= s_k_1, strides = st_1, kernel_initializer= i_k_1, activation = a_c_1))

    if number_layers !=1:

        list_2 = []
        st_2  = trial.suggest_categorical("strides2", [1,3,5,7])
        i_k_2 = trial.suggest_categorical("kernel_init2",['Orthogonal','lecun_uniform','he_normal'])
        a_c_2 = trial.suggest_categorical("activation_conv2",['relu','tanh','sigmoid'])
        
        for i in [16,32,64,128]:
            if i < s_f_1:
                list_2.append(i)
        s_f_2 = trial.suggest_categorical(f"filter_size2_{list_2}", list_2)
        model.add(Conv1D(filters= s_f_2, kernel_size= 1, strides = st_2, kernel_initializer= i_k_2, activation = a_c_2))

        list_3 = []
        st_3  = trial.suggest_categorical("strides3", [1,3,5,7])
        i_k_3 = trial.suggest_categorical("kernel_init3",['Orthogonal','lecun_uniform','he_normal'])
        a_c_3 = trial.suggest_categorical("activation_conv3",['relu','tanh','sigmoid'])
        
        for i in [8,16,32,64,128]:
            if i < s_f_2:
                list_3.append(i)
        s_f_3 = trial.suggest_categorical(f"filter_size3_{list_3}", list_3)
        model.add(Conv1D(filters= s_f_3, kernel_size= 1, strides = st_3, kernel_initializer= i_k_3, activation = a_c_3))

    model.add(GlobalMaxPooling1D())

    dense_layers = trial.suggest_categorical("n_dense_layers", [0,1,2])
    rate_drop= trial.suggest_categorical("drop_layer", [0,0.25,0.5])

    if rate_drop!=0:
        model.add(Dropout(rate= rate_drop))

    if dense_layers == 2:
        model.add(Dense(64,activation= trial.suggest_categorical("act_1d_layer", ['relu','tanh'])))# applies to all

    if dense_layers==2 or dense_layers==1:
        model.add(Dense(16,activation= trial.suggest_categorical("act_2d_layer", ['relu','tanh'])))# applies to all

    # sigmoid for binary
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)),
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=trial.suggest_categorical("batch_size", [128,256,512]),
              nb_epoch=EPOCHS,
              verbose=2,
              validation_data=(X_test, Y_test))

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(X_test, Y_test, verbose=2)

    with open('{}_{}.pickle'.format(rep,trial.number), 'wb') as fout:
        pickle.dump(model, fout)

    name_m= '{}_{}.h5'.format(rep,trial.number)
    model.save(name_m)

    return score[1]


if __name__ == "__main__":
    warnings.warn(
        "Recent Keras release (2.4.0) simply redirects all APIs "
        "in the standalone keras package to point to tf.keras. "
    )

    study_name = '{}_{}_{}_{}'.format(NN, approach, dataset, rep) # Unique identifier of the study.

    study = optuna.create_study(load_if_exists= True,direction="maximize", study_name=study_name, storage='sqlite:///{}_{}_{}_{}.db'.format(NN, approach, dataset, rep))
    study.optimize(objective, n_trials=25)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
