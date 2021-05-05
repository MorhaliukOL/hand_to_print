# hand_to_print
Convert handwriting to digital text

## Objective
Create robust Optical character recognition model with focus on Handwritten text recognition, with support of English and Russian languages.
And deploy as web application with Flask.

## Base model
Initial model is based on [this article](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5). <br>
Model architecture: <br>
&nbsp;&nbsp; 5 CNN layers <br>
&nbsp;&nbsp; 2 LSTM layers <br>
with CTC loss <br>

Model was trained on [IAM dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). <br>
Achieves character error rate of 10%.
