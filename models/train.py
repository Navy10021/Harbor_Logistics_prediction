import pandas as pd
from models.transformers import *


# 1. Load meta dataset
df = pd.read_csv("./data/meta_data(cargo).csv", index_col = "date")


# 2. Get features (from feature_selection.py)
selected_features = ['Busan', 'Ulsan', 'Samcheok', 'Gunsan', 'Gwangyang', 'Masan', 'Seogwipo']
series = df[selected_features]
target = "Busan"


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
    prediction_steps = 6,       # Predict month : 6 Months
    )
