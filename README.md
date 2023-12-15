# ssn-simulator
Code for simulating the Stabilized Supralinear Network in a visual perceptual learning task

1. _SSN_classes_middle_ and _SSN_classes_superficial_ contain the code for the corresponding SSN layers. If no orientation map is input to the middle layer, it will generate a new one. 
2. _model.py_ contains the functions used to run the model, both for the orientation discrimination task and to obtain the response for a single stimuli. When training with the orientation discrimination task, the parameters that you want to be trained regarding the SSN layer need to be included in _ssn_layer_pars_. Others need to be added to _constant_pars_. The code for the orientation discrimination task can be parallelized using _vmap_: the _in_axes_ variable of the vmap_function must match the _orientation_discrimination_ inputs. 
3. To train the model:
    3.1 Specify all the parameters in _parameters.py_
    3.2 Run _script_training.py_ where you need to specify the directory to output the results. This script calls on _training.py_ for normal training using accuracy, or on _training_staircase.py_ for staircase training.
    3.3 The output of training is a csv file containing the values for the trained parameters across epochs, and figures relating to the losses, accuracy, parameter changes, evolution of w_sig during training. 