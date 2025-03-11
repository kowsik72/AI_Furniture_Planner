# Furniture Arrangement AI

This project uses AI to generate optimized 2D furniture layouts based on user-defined room size and furniture selection.

## 🚀 Features
- Accepts custom **room dimensions** (Width × Height)
- Allows users to **select furniture** and specify dimensions
- Uses a **trained AI model** to predict optimal placements
- **Deploys as a REST API** using Flask
- **Visualizes** the layout using Matplotlib

---

## 📂 Project Structure
```
Furniture_Arrangement_AI/
│-- data_generator.py  # Generates training dataset
│-- model_train.py  # Trains AI model for furniture placement
│-- app.py  # Deploys AI model as an API
|-- furniture_dataset.csv # dataset
|-- furniture_model.pth # model 
│-- client.py  # Sends user input to API and retrieves layout
│-- README.md  # Project Documentation
```

---

## 🔧 Installation & Setup
### 1️⃣ Install Dependencies
```sh
pip install numpy pandas torch flask matplotlib rectpack requests
```
### 2️⃣ Generate Dataset
```sh
python data_generator.py
```
### 3️⃣ Train the Model
```sh
python model_train.py
```
### 4️⃣ Run the API Server
```sh
python app.py
```
The server runs on `http://localhost:5000`

### 5️⃣ Test with Client
Run the client to send a test request:
```sh
python client.py
```

---

## 📺 Video Demo
👉 **[Upload your 2-3 min demo video and add the link here]**

---

## 🤝 Contributors
- **kowsik** (kowsik72)

---


## 🌟 GitHub Repository
🔗 **[GitHub Repo Link](https://github.com/YOUR_USERNAME/Furniture_Arrangement_AI](https://github.com/kowsik72/AI_Furniture_Planner)**

