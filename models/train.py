import pandas as pd
from models.transformers import *


# 1. Load meta dataset
df = pd.read_csv("./data/meta_data(cargo).csv", index_col = "date")


# 2. Get features (from feature_selection.py)
target = "Busan"
#target = "Ulsan"
#target = "Gwangyang"
#target = "Mokpo"
#target = "Eastsea"
#target = "Pyeongtaek"
#target = "Jeju"

# dictionary we have already created from feature_selection.py
selected_features = {
    "Busan" : ['Busan', 'Ulsan', 'Gwangyang'],
    "Ulsan" : ['Ulsan', 'Okgye', 'Samcheok', 'Masan', 'Daesan', 'Gunsan', 'Gwangyang'],
    "Gwangyang" : ['Ulsan', 'Okgye', 'Samcheok', 'Eastsea', 'Gwangyang'],
    "Mokpo" : ['Jeju', 'Wando', 'Okpo', 'Mokpo', 'Eastsea'],
    "Eastsea" : ['Jeju', 'Mokpo', 'Eastsea'],
    "Jeju" : ['Jeju', 'Mokpo'],
    "Pyeongtaek" : ['Pyeongtaek', 'Eastsea']
}
series = df[selected_features[target]]



# 3. Prepare for my Transformers training.
train_data, val_data = get_data(series, target)
model = MyTransformer().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5, eps = 1e-7, weight_decay = 1e-3) # L2-Norm
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1.0, gamma = 0.1)  


# 4. Print prediction & evaluation graph per 100 epochs
train_eval(
    train_data = train_data,
    val_data = val_data,
    model = model,
    criterion = criterion,
    optimizer = optimizer,
    scheduler = scheduler,
    epochs = 8000,              # Best epochs : 5000 ~
    prediction_steps = 6,       # Months to predict : 6 Months
    )
