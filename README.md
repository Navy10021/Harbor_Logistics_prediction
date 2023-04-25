![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=300&section=header&text=%20ROK%20Port%20Logistics%20Forecast&fontColor=317589&fontSize=60)


## Neural Network(Transformers)-based Port Logistics Prediction Model 

### 1. Project Introductions

Comming Soon...

#### Dataset
**National Logistics Information Center** provides publicly cargo and passenger transportation data for 27 major ports in Korea (www.nlic.go.kr). We applied that data to our preprocessing process and then used it to analyze time series data and train(or predict) neural network models. 

#### Overall Pipeline
To summarize the entire process of the **『Transformers-based Port Logistics Prediction Model』** we designed, it consists of the following four steps.
  - STEP 1) Port logistics data pre-processing
  - STEP 2) Time series data analysis and visualization
  - STEP 4) Feature selection
  - STEP 3) Transformer-based model training and future port logistics prediction

+ IMG

### 2. Model Description

Transformers-based model architecture

+ IMG

### 3. Model Usage

#### STEP 1. Port logistics Data Pre-processing
The first step is to pre-process raw data and create time-series-based meta data in that 'data' directory by aggregating pre-processed data for each port. Then, it is ready to apply our data analysis tools and train our Transformers-based prediction model.
```python
$ python models/preprocessing.py
```

#### STEP 2. Time series Data Analysis and Visualization
In this step, the meta data converted to time series data is automatically analyzed with our time series data analysis tool. Executing the Python code below shows the results of time series data analyzed with three approaches: 1) Trend 2) Seasonality 3) Serial.
 ```python
$ python models/analysis_tool.py
```
If you want to get the analysis result of other port, just put the port name in target variable. ```target = 'Busan' ```. Default is 'Busan' port.

#### STEP 3. Feature selection

#### STEP 4. Transformer-based model training and future port logistics prediction

#### Prediction results.


### 4. Dev
  - Seoul National University GSDS NLP Labs
  - Busan National University BigData Labs
  - Navy Lee & Min-seop Lee
