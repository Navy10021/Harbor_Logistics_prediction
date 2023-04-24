import pandas as pd
import numpy as np
from datetime import datetime
import os

# 1. Get the collected Harbor Logistics Dataset (from National Logistics Information Center / 국가물류통합정보센터)
busan = pd.read_csv("./data/화물수송(부산).csv", encoding='utf-8')
ulsan = pd.read_csv("./data/화물수송(울산).csv", encoding='utf-8')
pohang = pd.read_csv("./data/화물수송(포항).csv", encoding='utf-8')
pyeongtaek = pd.read_csv("./data/화물수송(평택당진).csv", encoding='utf-8')
tongyeong = pd.read_csv("./data/화물수송(통영).csv", encoding='utf-8')
taean = pd.read_csv("./data/화물수송(태안).csv", encoding='utf-8')
jinhae = pd.read_csv("./data/화물수송(진해).csv", encoding='utf-8')
jeju = pd.read_csv("./data/화물수송(제주).csv", encoding='utf-8') 
janghang = pd.read_csv("./data/화물수송(장항).csv", encoding='utf-8')
geoje = pd.read_csv("./data/화물수송(장승포).csv", encoding='utf-8')
wando = pd.read_csv("./data/화물수송(완도).csv", encoding='utf-8')
okpo = pd.read_csv("./data/화물수송(옥포).csv", encoding='utf-8')
okgye = pd.read_csv("./data/화물수송(옥계).csv", encoding='utf-8')
yeosu = pd.read_csv("./data/화물수송(여수).csv", encoding='utf-8')
sokcho = pd.read_csv("./data/화물수송(속초).csv", encoding='utf-8')
seogwipo = pd.read_csv("./data/화물수송(서귀포).csv", encoding='utf-8')
samcheonpo = pd.read_csv("./data/화물수송(삼천포).csv", encoding='utf-8')
samcheok = pd.read_csv("./data/화물수송(삼척).csv", encoding='utf-8')
boryeong = pd.read_csv("./data/화물수송(보령).csv", encoding='utf-8')
mokpo = pd.read_csv("./data/화물수송(목포).csv", encoding='utf-8')
masan = pd.read_csv("./data/화물수송(마산).csv", encoding='utf-8')
eastsea = pd.read_csv("./data/화물수송(동해묵호).csv", encoding='utf-8')
daesan = pd.read_csv("./data/화물수송(대산).csv", encoding='utf-8')
gunsan = pd.read_csv("./data/화물수송(군산).csv", encoding='utf-8')
gwangyang = pd.read_csv("./data/화물수송(광양).csv", encoding='utf-8')
gohyun = pd.read_csv("./data/화물수송(고현).csv", encoding='utf-8')
gyeongin = pd.read_csv("./data/화물수송(경인).csv", encoding='utf-8')


# 2. Preprocessing function
def preprocessing(dataframe, port_name):
    # Total Harbor  
    dataframe = dataframe[dataframe["외내항구분"] == "총계"]
    dataframe = dataframe.drop(columns = ["외내항구분", "국적선구분", "KF_CD", "ED_CD"])
    dataframe = dataframe.rename(columns = {"조회년도" : "year", "조회월" : "month", "입출항구분": "ship", "MEASURE" : port_name})

    # Drop & Create Columns(2000.1. ~ 2023.2.)
    dataframe['year']= dataframe['year'].astype('str')
    dataframe['month']= dataframe['month'].astype('str')
    for i in range(1, 10):
        dataframe = dataframe.replace({'month' : str(i)}, '0'+str(i))
    dataframe['date'] = dataframe['year'] + dataframe['month'] 
    dataframe['date'] = dataframe['date'].apply(lambda _ : datetime.strptime(_, "%Y%m"))
    dataframe['date'] = pd.to_datetime(dataframe['date'], utc = False)
    dataframe = dataframe.drop(columns = ["year", "month"])
    dataframe = dataframe[['date', port_name, 'ship']]
    dataframe = dataframe.set_index(keys = ['date'], drop = True)
    #dataframe["port"] = port_name
    dataframe = dataframe[dataframe.index > "1999-12-01"]

    # Two tpye of dataset : Ferry and Cargo
    df_ferry = dataframe[dataframe["ship"] == "연안여객선"]
    df_ferry = df_ferry.drop(columns = ["ship"]) 
    df_cargo = dataframe[dataframe["ship"] == "연안화물선"]
    df_cargo = df_cargo.drop(columns = ["ship"])

    return df_ferry, df_cargo
  
  
# 3. Create hash map
meta_dict = {
    "busan" : busan,
    "ulsan":ulsan, 
    "pohang":pohang,
    "pyeongtaek":pyeongtaek,
    "tongyeong":tongyeong,
    "taean":taean,
    "jinhae ":jinhae, 
    "jeju":jeju,
    "janghang":janghang,
    #"geoje":geoje,
    "wando":wando, 
    "okpo":okpo, 
    "okgye":okgye, 
    "yeosu":yeosu,
    "sokcho":sokcho,
    "seogwipo":seogwipo, 
    "samcheonpo":samcheonpo, 
    "samcheok":samcheok, 
    "boryeong":boryeong, 
    "mokpo":mokpo, 
    "masan":masan, 
    "eastsea":eastsea, 
    "daesan":daesan, 
    "gunsan":gunsan, 
    "gwangyang":gwangyang, 
    "gohyun":gohyun, 
    #"gyeongin":gyeongin, 
             }


# 4. Concate ALL dataset
new_meta_dict = dict()
for key, val in meta_dict.items():
    new_meta_dict[key[0].upper()+key[1:]] = val

meta_dict = new_meta_dict
meta_cargo, meta_ferry = [], []

for i in meta_dict:
    ferry, cargo = preprocessing(meta_dict[i], i)
    meta_ferry.append(ferry)
    meta_cargo.append(cargo)
    print(" {} data size : {}".format(i, len(cargo)))

cargo_df = pd.concat(meta_cargo, axis=1)
ferry_df = pd.concat(meta_ferry, axis=1)
print("\n >> Complete concatenation of all datasets.")


# 5. Fill in missing values with zero values
cargo_df = cargo_df.fillna(0.0)
ferry_df = ferry_df.fillna(0.0)
cargo_df = cargo_df.astype({"Ulsan":"float","Jeju":"float", "Yeosu":"float", "Mokpo":"float", "Masan":"float", "Eastsea":"float", "Daesan":"float"})
ferry_df = ferry_df.astype({"Ulsan":"float","Jeju":"float", "Yeosu":"float", "Mokpo":"float", "Masan":"float", "Eastsea":"float", "Daesan":"float"})
print(" >> Finish filling in the missing data.")


# Check missing values
print(" >> Cargo dataset rows : ", len(cargo_df))
print(" >> MISSING VALUES ?", cargo_df.isna().sum().sum())
print("==============================")
print(cargo_df.isna().sum())
print("==============================")
#print(">> Ferry dataset rows : ", len(ferry_df))
#print(">> MISSING VALUES ?", ferry_df.isna().sum().sum())
#print("=============================")
#print(ferry_df.isna().sum())


# 6. Save the meta dataset 
cargo_df.to_csv("./data/meta_data(cargo).csv")
ferry_df.to_csv("./data/meta_data(ferry).csv")


print("/n >> Data preprocessing and saving meta-dataset are Done.")
