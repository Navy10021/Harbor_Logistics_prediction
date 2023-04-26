![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=300&section=header&text=%20ROK%20Port%20Logistics%20Forecast&fontColor=317589&fontSize=60)


## Neural Network(Transformers)-based Port Logistics Prediction Model 

### 1. Project Introductions

Comming Soon...

#### Dataset
**National Logistics Information Center** provides publicly cargo and passenger transportation data for 27 major ports in Korea (www.nlic.go.kr). We applied that data to our preprocessing process and then used it to analyze time series data and train(or predict) a neural network model. 


### 2. Model Description

#### Overall Pipeline
To summarize the entire process of the **『Transformers-based Port Logistics Prediction Model』** we designed, it consists of the following four steps.
  - STEP 1) Port logistics data pre-processing
  - STEP 2) Time series data analysis and visualization
  - STEP 3) Feature selection
  - STEP 4) Transformer-based model training and future port logistics prediction

+ IMG

#### Transformers-based Model

The **Transformers model** solves the problems faced by the existing RNN-based models by applying the **attention mechanism**, and the calculation speed is greatly improved.  In particular, attention is a core concept of the Transformers, which enables the neural network of the model to understand contextual text information, focusing on words similar to the current token, and training and inferencing. Inspired by this original model, our **『Transformers-based Port Logistics Prediction Model』** is a **Seq2Seq model** consisting of an encoder network as **three Transformers encoders are stacked** and a decoder network as **two layers MLP(Multi-Layer Perceptron)** using a **GULE**(Gaussian Error Linear Units) activation function.

![my_transformers](https://user-images.githubusercontent.com/105137667/234526953-1165f18c-b57a-4979-abad-bda6c8af7f9e.jpg)

### 3. Model Usage

#### STEP 1. Port logistics Data Pre-processing
The first step is to pre-process raw data and create **time series-based metadata** in that 'data' directory by aggregating pre-processed data for each port. Then, it is ready to apply our time series data analysis tools and train our Transformers-based prediction model.

```python
$ python models/preprocessing.py
```

#### STEP 2. Time series Data Analysis and Visualization
In this step, the metadata converted to time series data is automatically analyzed with our **time series data analysis tool**. Executing the Python code below shows the results of time series data analyzed with three approaches: **1) Trend, 2) Seasonality, 3) Serial**.

 ```python
$ python models/analysis_tool.py
```

If you want to get the analysis result of other ports, just put the port name in target variable ```target = 'Busan' ```. Default is 'Busan' port.

#### STEP 3. Feature selection
This step returns any features (or columns) highly related to the target's port logistics. The Python code below **provides optimal features** based on **1) Pearson's correlation coefficient** and **2) Ensemble learning methods (XGBoost)**.

 ```python
$ python models/feature_selection.py
```

#### STEP 4. Transformer-based model training and future port logistics prediction
The final step is to import ```transformers.py ``` code to train the Transformer-based model and predict future port traffic. The results predicted by the model are shown as a plot graph per each iteration the user sets.

 ```python
$ python models/train.py
```

#### Prediction results.

+ IMG

### 4. Dev
  - Seoul National University GSDS NLP Labs
  - Busan National University BigData Labs
  - Navy Lee & Min-seop Lee
