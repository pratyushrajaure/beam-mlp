# Beam Deflection Prediction using MLP



A PyTorch-based Multi-Layer Perceptron (MLP) that predicts the deflection of a simply supported beam subjected to a central point load.



## Features



i) Generate synthetic beam deflection data.



ii) Split data into training, validation, and testing sets.



iii) Normalize input and output features.



iv) Train an MLP using PyTorch, Adam optimizer, and MSE loss.



v) Evaluate predictions using MAE, RMSE, and Relative L2 Error.



vi) Generate plots to visualize model performance.



## Model Architecture



`5 → 64 → 64 → 1`



ReLU activation is used in the hidden layers.



## Project Structure



## Project Structure

```text
.
├── beams.py
├── generate_data.py
├── train.py
├── beam_data.csv
├── best_model.pth
└── results/
    ├── learning_curve.png
    ├── predicted_vs_true.png
    ├── error_histogram.png
    └── three_beam_curves.png

```



## Requirements



i) Python



ii) NumPy



iii) Pandas



iv) Matplotlib



v) PyTorch



Install dependencies:



```bash

python -m pip install -r requirements.txt

```



## Usage



```bash

python generate\_data.py

python train.py

```



The trained model, evaluation metrics, and visualization plots are generated during execution.

