# ML Inductor QLR Predictor
***
*This Repo is the outcome of an ongoing Master Degree's research project in EEE at UCC, in collaboration with Tyndall National Institute*

## *Accuracy*
With the following parameter:  
Trainer: `epochs = 2000, alpha = 1.0, beta = 6.0`  
Data_splitter: `proportion = 0.7`  
The lowest MPE tested on the test dataset is:  
* ***R: 1%***  
* ***L: 0.95%***  
* ***Q: 2.7%%***

## Data process
Put all data in `RLQ` Folder, or change the initial parameter of `data_processor()` in `main.py`
Data_processor will automatically read filenames and data, then generate `data.csv` in root folder.

## Machine Learning Model

### PINN Model
The model contains 3 hHidden Layers, 2 Layer Normalization, with 7 inputs and 2 outputs.  
The activation function is `Tanh()`.  
Here's the sequence of the model.  
`[Input 6(without freq)] -> [H1, 32 nodes] -> Layer Normalization -> [add freq data] -> [H2, 16 nodes] -> Layer Normalization -> [H3, 8 nodes] -> [Output 2]`

### Training Parameters:
The model introduced physical information to improve the accuracy of prediction.  
Total `loss` is based on 2 parts, MSE of prediction, and MSE of Q.  
Q is calculated from the equation:  
*Q = 2 pi * f * L / R*
* `alpha`: coefficient of Prediction MSE, default = 1.0
* `beta` : coefficient of Q MSE, default = 6.0

## Train and test Model
run `main.py`