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



```text

├── beams.py

├── generate\_data.py

├── train.py

├── beam\_data.csv

├── best\_model.pth

└── results/

&#x20;   ├── learning\_curve.png

&#x20;   ├── predicted\_vs\_true.png

&#x20;   ├── error\_histogram.png

&#x20;   └── three\_beam\_curves.png

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

