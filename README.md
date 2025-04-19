<!-- Logo of website -->
<div align="center">

  <img src="https://miro.medium.com/v2/resize:fit:1000/1*yrgbW7GvOcp94f-5HZcmyQ.jpeg" width="500">

</div>

<!-- Introduction of project -->

<div align="center">

# 50.039 Theory and Practice of Deep Learning

</div>

<h2 align="center" style="text-decoration: none;">
Deep Learning Models for Time Series Data using Seq2Seq RNN and Transformer
</h2>

## Project Overview :

Accurate weather forecasting plays a critical role in supporting urban planning, agriculture, public health, disaster preparedness, and transportation systems—particularly in highly populated cities like Delhi. However, the inherently dynamic and complex nature of weather patterns, influenced by both natural phenomena and human activities, makes precise forecasting a challenging task. In this project, we aim to improve the accuracy of short-term weather prediction in Delhi using deep learning. We train and evaluate multiple architectures, including a Seq2Seq RNN model, a Transformer-based model, and pre-trained time series models, on daily weather data collected between January 1, 2013, and April 24, 2017. The dataset includes four key features: average temperature, humidity, wind speed, and average air pressure. These models are used to predict future values of the four variables over the following months. Their performance is evaluated by comparing predictions to real recorded values, allowing us to assess model accuracy and robustness. The goal is to develop a reliable weather forecasting tool that can aid households, organizations, and government authorities in proactive decision-making and risk mitigation.

## Contributors :

- Louis Anh Tran
- Shaoren Ong
- Benetta Cheng

## Model Architectures:

- [x] **Seq2Seq auto-regressive** Recurrent Neural Network using **Long Short-Term Memory (LSTM)**
- [x] **Seq2Seq non-auto-regressive** Recurrent Neural Network using **Long Short-Term Memory (LSTM)**
- [x] **Seq2Seq auto-regressive** Recurrent Neural Network using **Gated Recurrent Unit (GRU)**
- [x] **Seq2Seq non-auto-regressive** Recurrent Neural Network using **Gated Recurrent Unit (GRU)**
- [x] **Seq2Seq Transformer** architecture

## Main components:

- Time series dataset for training and evaluation is placed under [Time series dataset](https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data/tree/main/dataset)
- Data exploration and visualization is placed under [Dataset exploration](https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data/blob/main/VizData_Notebook.ipynb)
- Models training detailed process and steps for all architectures are placed under [Models training](https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data/tree/main/models_training)
- Models performance evaluation and comparision are consolidated and placed under  [Consolidated models evaluation and comparison](https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data/blob/main/Conso_VizEval_Notebook.ipynb)
- Models Pytorch weights for reproducibility are placed under  [Models Weight for reproducibility](https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data/tree/main/models_weights_storage)
- Figures for Loss curve during model training are placed under [Loss curves during training](https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data/tree/main/images_model_training_and_eval)
- Model architectures are defined and placed under [Model architectures definition](https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data/tree/main/model_architecture_definition)
- Hyperparameters for all model architectures are placed under [Configurations](https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data/tree/main/config)
- Helper functions are defined and placed under [Helpers functions](https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data/tree/main/utils)

## Built With

This section outlines the packages used in the project

- torch
- torchvision
- matplotlib
- seaborn
- tensorboard
- plotly
- pandas
- scikit-learn
- tqdm
- opencv-python

Refer to [requirements.txt](https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data/tree/main/models_training) file for your references

### Getting started

1. Clone the project repo:

```
git clone https://github.com/LouisAnhTran/deep_learning_models_for_time_series_data.git
```

2. Run the following commands to install required packages and set up virtual environment:

```
# Change to project main directory
cd deep_learning_models_for_time_series_data

# Set up Python virtual environment
python -m venv venv && source venv/bin/activate

# Make sure PIP is up to date
pip install -U pip wheel setuptools

# Install required dependencies
pip install -r requirements.txt
```


## How to Navigate the Project Codebase (Recommended Order)

1. **`VizData_Notebook.ipynb`** – Provides in-depth exploration and analysis of the dataset, including statistics, size, number of features, distributions, and key properties.

2. **`config/config_LSTM.py`** – Contains all hyperparameters required to train the model using the LSTM architecture. Feel free to modify any parameter as needed.

3. **`models_training/seq2seq_autoregressive_LSTM_final.ipynb`** – Includes detailed steps for training a model from scratch using the training set, as well as evaluation on the test set. This notebook also saves the PyTorch model weights and training loss curves for reference.

4. **`images_model_training_and_eval/`** – Stores all training loss curve plots. Choose the image that corresponds to the model you are working with.

5. **`models_weights_storage/`** – Contains the trained PyTorch model weights.

6. **`model_architecture_definition/`** – Includes the Python classes that define model architectures. If you modify a model’s structure in a notebook, make sure to update the corresponding class here.

7. **`Conso_VizEval_Notebook.ipynb`** – Contains the code to reproduce results using model saved weights, visualize training loss curves, and evaluate model performance for all models. It also includes comparative analysis across all trained models and architectures.

**`Note`**: As this project explores diverse model architectures, the flow above is exemplified using the Seq2Seq LSTM model. However, the same structure applies to other model architectures as well. You can follow the same described flow to explore and evaluate any of the implemented models.



<!-- ACKNOWLEDGMENTS -->
## References:

- [Pytorch](https://pytorch.org/)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png


[fastapi-shield]: https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi
[streamlit-shield]: https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white
[langchain-shield]: https://img.shields.io/badge/LangChain-ffffff?logo=langchain&logoColor=green




