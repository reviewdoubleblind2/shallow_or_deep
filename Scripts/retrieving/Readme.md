
Here you can find two scripts for retrieving the best model:
* **retrieve_best_CNN_RNN.py:** you can use this script for obtaining the best model after the process of hyper-tuning  for CNN and RNN with Optuna. 
* **retrieve_best_RF.py :** you can use this script for obtaining the best model after the process of hyper-tuning  for Rando forest with Optuna. 

If you want to obtain the best model you can change the following constants for:

**retrieve_best_CNN_RNN.py**:

* **APP** can be 'binary' and 'multi' 
* **DATA_S** can be 'Russell' (R-DS), 'Juliet' (J-DS) and 'OUR' (GH-DS)
* **REP_D** can be 'b0' (R0), 'b1' (R1), 'b1_int' (R2), 'b1_iden' (R3), 'b1_int_iden' (R4)
* **NN**  can be 'CNN' and 'RNN'


**retrieve_best_RF.py**:

* **approach** can be 'binary' and 'multi' 
* **dataset** can be 'Russell' (R-DS), 'Juliet' (J-DS) and 'OUR' (GH-DS)
* **rep** can be 'b0' (R0), 'b1' (R1), 'b1_int' (R2), 'b1_iden' (R3), 'b1_int_iden' (R4)
* **NN**  can be 'RF'

