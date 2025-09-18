# AI-Data-Storyteller
AI-powered data storytelling application that automates exploratory data analysis, generates plain-English insights, and creates meaningful visualizations. Built with Python, Pandas, Seaborn, and Streamlit, it delivers an interactive dashboard with automated reporting for data-driven decision-making.


# 📊 Data Storyteller (Streamlit Dashboard)

# Overview

**Data Storyteller** is a **Streamlit web application** built to simplify **Exploratory Data Analysis (EDA)** and reporting. With just a CSV upload, the tool generates:

* A **structured dataset summary**
* **Key insights and metrics**
* **Interactive visualizations** for deeper exploration
* **Exportable reports** in PDF and Word formats

This project focuses on **automation, interactivity, and usability** for analysts and business users who need quick insights from raw datasets.


# Key Features

**CSV Upload** – Upload datasets directly from the dashboard.
**Automated EDA** – Provides dataset size, missing values, and descriptive statistics.
**Insight Summaries** – Organized into overviews, highlights, and recommendations.
**Interactive Visuals** – Histograms, bar charts, and heatmaps powered by Plotly.
**Report Export** – Generate professional reports in **PDF** or **DOCX** format.



# 📂 Project Structure

```
Project_Folder/
├── code/                # Optional: initial EDA scripts or notebooks
├── dashboard/
│   └── app.py           # Streamlit application
├── report/
│   ├── summary.pdf      # Sample generated PDF report
└── README.md
```

---

### Installation & Setup

# Prerequisites

* Python **3.9+**

# 📥 Install Dependencies

Clone the repository and install requirements:

```bash
pip install -r requirements.txt
```

---

## Running the Application

1. Navigate to the `dashboard` folder.
2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```
3. Open the link in your browser to upload a dataset and explore the results.

---

## 🛠️ Tech Stack

* **Python** – Pandas, NumPy
* **Visualization** – Matplotlib, Seaborn, Plotly
* **Dashboard** – Streamlit
* **Reporting** – FPDF / python-docx

---

## 📌 Use Cases

* Quick **EDA automation** for data analysts
* Fast **business reporting** with visual summaries
* **Data-driven decision support** without manual effort
* Suitable for **academic and professional projects**

---

## 📜 License

This project is open-source and available under the **MIT License**.
