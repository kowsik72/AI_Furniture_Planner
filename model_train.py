import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import os

# Define the model with dynamic input/output sizes
class FurnitureModel(nn.Module):
    def __init__(self, input_size=None, output_size=None):  # Fixed init method
        super(FurnitureModel, self).__init__()
        if input_size is None or output_size is None:
            raise ValueError("input_size and output_size must be provided")
        self.fc1 = nn.Linear(input_size, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train the model
def train_model(X_train, y_train, X_test, y_test, epochs=100):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    print(f"Initializing model with input_size={input_size}, output_size={output_size}")
    
    model = FurnitureModel(input_size=input_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_preds = model(X_train_tensor).numpy()
                test_preds = model(X_test_tensor).numpy()
                train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
                test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
    
    return model, train_preds, test_preds

if __name__ == "__main__":  # Fixed the typo
    # Check if dataset exists
    dataset_path = "furniture_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset file `{dataset_path}` not found!")
        exit(1)
    
    print("✅ Loading dataset...")
    df = pd.read_csv(dataset_path)
    X = np.array(df["input"].apply(eval).tolist())
    y = np.array(df["output"].apply(eval).tolist())

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save test data for evaluation in app.py
    test_df = pd.DataFrame({"input": [x.tolist() for x in X_test], "output": [y.tolist() for y in y_test]})
    test_df.to_csv("furniture_test_dataset.csv", index=False)
    print(f"✅ Saved test dataset to `furniture_test_dataset.csv` with {len(X_test)} samples.")

    # Train model
    print("🚀 Training model...")
    model, train_preds, test_preds = train_model(X_train, y_train, X_test, y_test)
    
    # Save model
    model_save_path = "furniture_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model trained and saved as `{model_save_path}`.")
