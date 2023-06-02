![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=300&section=header&text=%20%20Harbor%20Logistics%20Forecaster&fontColor=317589&fontSize=60)


## Transformers-based Neural Network Harbor Logistics Prediction Model 


## 1. Project Introductions

The purpose of this project is to analyze the status of ship departures by 27 major domestic ports and to predict future port logistics by leveraging AI modeling.
To this end, we built a **ⓐ「Time-series data analysis process」** optimized for port logistics and a unique **ⓑ「Transformers-based Neural Network Prediction Model」**. The **ⓐ「Time series data analysis process」** designed by us preprocesses the raw port logistics dataset and then automatically visualizes the results of big data analysis based on trend, seasonality and serial. In addition, the **ⓑ「Transformer-based Neural Network Prediction Model」** we created, inspired by the latest models applied to the field of natural language processing (NLP) such as GPT (Generative Pre-trained Transformers) and BERT (Bidirectional Encoder Representations from Transformers), is a model that predicts numerical data. This model trains on the port logistics dataset and predicts future port traffic volume. According to the evaluation metrics, this model shows excellent predictive performance. In the future, we hope that the big data analysis process and prediction model we designed will be expanded and applied to various fields.


이 프로젝트의 목적은 AI 기반 모델링을 활용하여 국내 27개 주요 항만별 선박출입항 현황 데이터를 분석하고, 미래의 물동량을 예측하는데 있다. 이를 위해 우리는 항만 물류에 최적화된 **ⓐ「시계열 데이터 분석 프로세스」** 와 고유의 **ⓑ「트랜스포머 기반 신경망 예측모델」** 을 구축하였다. 우리가 디자인한 **ⓐ「시계열 데이터 분석 프로세스」** 는 항만 물류 데이터를 전처리(Pre-processing)한 다음 핵심 데이터를 선택(Feature selection)해주며, 시간순으로 재구성된 데이터의 트랜드(Trend), 계절성(Seasonality), 계열성(Serial)을 전문가 수준으로 분석한 결과를 제공한다. 또한, GPT(Generative Pre-trained Transformers)  및 BERT(Bidirectional Encoder Representations from Transformers)와 같은 자연어처리(NLP) 분야에 적용되고 있는 최신 모델들에서 영감을 얻어 본 연구실이 생성한 **ⓑ「트랜스포머 기반 신경망 예측모델」** 은 연속적인 데이터를 학습하고 예측하는 seq2seq 모델로, 주어진 항만 물류 데이터를 학습하고 미래의 물동량을 예측한다. 평가 메트릭에 따르면 이 모델은 우수한 예측 성능을 보여준다. 향후, 우리가 디자인한 시계열 빅데이터 분석 프로세스와 예측 모델이 다양한 분야에 확대 적용되길 기대해 본다.


### Dataset
**National Logistics Information Center** provides publicly cargo and passenger transportation data for 27 major ports in Korea (www.nlic.go.kr). We applied that data to our preprocessing process and then used it to analyze time series data and train(also predict) a Transfomer-based neural network model. 


## 2. Model Description

### Overall Pipeline
To summarize the entire process of the **『Transformers-based Port Logistics Prediction Model』** we designed, it consists of the following four steps.
  - STEP 1) Port logistics data pre-processing
  - STEP 2) Time series data analysis and visualization(model ①)
  - STEP 3) Feature selection
  - STEP 4) Transformer-based neural network model training and future port logistics prediction(model ②)

![overall](https://user-images.githubusercontent.com/105137667/235141521-1d2a0a20-a7a1-4287-8ab1-585b06f9b426.jpg)


### Transformer-based Neural Network Prediction Model

The **Transformers model** solves the problems faced by the existing RNN-based models by applying the **attention mechanism**, and the calculation speed is greatly improved.  In particular, attention is a core concept of the Transformers, which enables the neural network of the model to understand contextual text information, focusing on words similar to the current token, and training and inferencing. Inspired by this original model, our **『Transformers-based Port Logistics Prediction Model』** is a **Seq2Seq model** consisting of an encoder network as **three Transformers encoders are stacked** and a decoder network as **two layers MLP(Multi-Layer Perceptron)** using a **GELU**(Gaussian Error Linear Units) activation function.

![my_transformers](https://user-images.githubusercontent.com/105137667/234526953-1165f18c-b57a-4979-abad-bda6c8af7f9e.jpg)


## 3. Model Usage

#### STEP 1. Data Pre-processing
The first step is to pre-process raw data and create **time series-based metadata** in that 'data' directory by aggregating pre-processed data for each port. Then, it is ready to apply our time series data analysis tools and train our Transformers-based prediction model.

```python
$ python models/preprocessing.py
```

#### STEP 2. Time series Data Analysis Tool
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

#### STEP 4. Transformer-based Neural Network Model training and prediction
The final step is to import ```transformers.py ``` code to train the Transformer-based neural network model and predict future port traffic. The results predicted by the model are shown as a plot graph per each iteration the user sets.

 ```python
$ python models/train.py
```

## 4. Result visualization

#### ★ The evaluation result of our model against the validation dataset per training iterations ★ 
* Blue line : True , Red line : Model prediction, Gray line : Residual

![iteration_sum](https://github.com/Navy10021/Harbor_Logistics_prediction/assets/105137667/1cddce95-3a94-4c29-a61b-7de808c63d62)


#### ★ Predicted results of future logistics volume (6 months) in 6 major Korean ports ★ 

![port_prediction_sum](https://github.com/Navy10021/Harbor_Logistics_prediction/assets/105137667/8e849898-0b61-4f96-bc98-0bfbc65f102c)


## 5. Dev
  - Seoul National University GSDS NLP Labs
  - Busan National University BigData Labs
  - Navy Lee & Min-seop Lee
