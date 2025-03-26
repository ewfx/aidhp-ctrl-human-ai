# 🚀 AI-Driven Hyper-Personalization & Recommendations Dashboard  

## 📌 Table of Contents  
- [Introduction](#-introduction)  
- [Demo](#-demo)  
- [What It Does](#-what-it-does)  
- [How We Built It](#-how-we-built-it)  
- [Challenges We Faced](#-challenges-we-faced)  
- [How to Run](#-how-to-run)  
- [Tech Stack](#-tech-stack)  
- [Team](#-team)  

---

## 🎯 Introduction  
An AI-powered dashboard that transforms financial services by offering:  
✅ **Hyper-personalized recommendations** (credit cards, loans, investments)  
✅ **Predictive business insights** (customer segmentation, churn prediction)  
✅ **Multi-modal search** (text + image-based product recommendations)  

### 🌟 Addressing 3 Industry Challenges:  
❌ **Generic financial advice** → ✅ AI-driven personalized recommendations  
❌ **Manual customer segmentation** → ✅ Automated insights & clustering  
❌ **Lack of predictive analytics** → ✅ AI-powered early warnings  

---

## 🎥 Demo  
### 🖥️ Run Locally:  
```bash
point your terminal to src folder
src> pip install -r requirements.txt  
src> streamlit run app.py
```
## How We Built It

    A[CSV/Image Upload] --> B(Pandas Data Pipeline)  
    B --> C{AI Models}  
    C --> D[Sentiment Analysis]  
    C --> E[CLIP Visual Search]  
    C --> F[Churn Prediction]  
    C --> G[K-Means Clustering]  
    F --> H[Streamlit Dashboard]  
## What It Does
###1️⃣ Personal Recommendations
Suggests credit cards, loans, investments based on:

Income & credit profiling

Risk appetite analysis (⭐⭐⭐⭐⭐ Scoring)

###2️⃣ Business Intelligence
Customer Segmentation: Identify loyal customers & at-risk groups

Churn Prediction: AI-powered 6-month early warning system

###3️⃣ Multi-Modal Search
CLIP-powered visual product search

Find products using text + image querie

## Challenges We Faced
Challenge	Solution
Mixed data types in CSVs	Auto-conversion with pd.to_numeric()
Large CLIP model size	Optimized with ONNX runtime
Real-time scoring optimization	Cached Sklearn pipelines

## How to Run

Step 1 : Download and install Python 3.x from official website

Step 2 : Navigate to the project directory to AI-Driven-Hyper-Personilazation\code\src and Open a terminal or command prompt

Step 3 : Run the following commands:

```bash
Point your terminal to src folder
src> pip install -r requirements.txt  

src> streamlit run app.py

#Run Tests
python test-app.py

```
## Tech Stack
Component	Technology
Frontend	Streamlit
AI/ML	Transformers, Sklearn, PyTorch
Data	Pandas, NumPy
Deployment	Docker (Local), AWS EC2 (Prod)

## Ctrl-Human-AI Team
KRLX-25 - AI-Human Controlled System

Krishna/Kiranmai - Wisdom & Strategy
Rajesh/Ravi - Leadership & Resilience
Laxman - Stability & Intelligence
