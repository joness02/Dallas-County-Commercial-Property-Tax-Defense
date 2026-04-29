# Dallas County Commercial Tax Defence Agent

An AI-powered property tax protest defense system built for the Dallas County Appraisal District to help evaluate commercial property tax protests, predict the minimum legally defensible settlement floor, and support county appraisers in making faster, more consistent decisions.

This project combines **LightGBM Quantile Regression**, **SHAP Explainability**, and a **Groq-powered conversational AI agent** to automate tax protest analysis and reduce revenue loss caused by under-defended settlements.

---

## Project Overview

Every year, corporate law firms file thousands of bulk tax protests against high-value commercial properties in Dallas County.

Due to time constraints and case volume, the county is often forced to accept low settlement offers—not because valuations are incorrect, but because there is not enough time to manually defend every case.

This project solves that problem by building an intelligent defense agent that:

- Predicts the **minimum legally defensible settlement floor**
- Explains valuation decisions using **SHAP feature attribution**
- Supports both:
  - a full **desktop GUI interface**
  - a **conversational AI chatbot agent**
- Helps prevent unnecessary county revenue loss

As described in the project report, the system focuses on automating commercial property protest defense and improving ARB settlement decisions :contentReference[oaicite:0]{index=0}.

---

## Key Features

### Predictive Valuation Model

Uses **LightGBM Quantile Regression** to estimate a defensible floor rather than a simple market value prediction.

This ensures:

- conservative legal defense
- calibrated settlement thresholds
- operational decision support

The final model is calibrated to approximately the **10th percentile floor prediction**, which helps determine whether a lawyer's offer should be accepted or rejected :contentReference[oaicite:1]{index=1}.

---

### High-Value Specialist Model

Properties above:

$10,000,000

are treated separately using a specialist model for improved SHAP explanation quality.

This improves performance for institutional and complex commercial properties.

---

### SHAP Explainability

Top valuation drivers are shown clearly for every prediction, including:

* prior certified value
* building area
* effective age
* depreciation spread
* floor area ratio
* exemptions
* zoning/location factors

This allows appraisers to justify decisions clearly during ARB hearings.

---

### Desktop GUI Application

A full Tkinter-based interface for:

* entering property details
* reviewing model predictions
* viewing SHAP waterfall plots
* scenario analysis
* settlement decision support

File:

```bash
agent_gui.py
```

---

### Power BI Dashboard

An interactive Power BI dashboard was developed to provide a visual overview of commercial property valuation trends, model outputs, and protest decision insights.

The dashboard helps stakeholders quickly analyze:

- predicted defensible settlement floors
- lawyer offer comparisons
- accept vs reject decision patterns
- high-value property distribution
- key valuation drivers and trends
- neighborhood and ZIP-based property insights

This makes the project more practical for operational use by allowing appraisers and decision-makers to monitor patterns at scale rather than reviewing properties one by one.

---

### Conversational AI Agent

A Groq-powered legal defense chatbot that allows users to ask:

* Should we reject this protest?
* What if the offer increases?
* Why is this property valued this way?

The chatbot automatically calls prediction tools and responds with operational recommendations.

File:

```bash
agent_chat.py
```

---

## Tech Stack

### Machine Learning

* Python
* LightGBM
* SHAP
* Scikit-learn
* Pandas
* NumPy

### Interface

* Tkinter
* Matplotlib

### LLM Integration

* Groq API
* Llama 3.3 70B Versatile

---

## Project Structure

├── model/  
│   ├── cat_impute_values.pkl  
│   ├── categorical_features.pkl  
│   ├── feature_cols.pkl  
│   ├── final_model.pkl  
│   ├── hv_model.pkl  
│   ├── impute_values.pkl  
├── Final.ipynb    
├── agent_chat.py  
├── agent_gui.py  
├── README.md  
├── Dashboard.pbix

---

## Dataset

### Source

Dallas County Appraisal District (DCAD) commercial property appraisal data

The final training dataset was created by merging multiple appraisal-related tables including:

* account appraisal year
* account info
* exemption values
* commercial details
* land details
* taxable objects

After cleaning and preprocessing:

36,279 commercial property records
34 engineered features

were used for final modeling .

---

## Model Performance

### Main Model

### Quantile Regression Floor Model

Performance:

* Percentile hit target: ~10%
* Achieved: **9.7%**
* MAE significantly reduced after calibration

This ensures the predicted floor behaves as a legally defensible lower bound rather than a standard prediction.

---

### High-Value Specialist Model

For properties over $10M:

| Model               |  MAPE |
| ------------------- | ----: |
| Main Model          | 17.5% |
| HV Specialist Model | 12.8% |

Improvement:

4.7% better performance

The specialist model is used for SHAP attribution only, while the calibrated main model provides the final floor prediction .

---

## Example Decision Logic

### If:

Predicted Floor = $8,500,000
Lawyer Offer    = $7,900,000

### Result:

DECISION: REJECT

Because the offer falls below the defensible settlement floor.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/joness02/Dallas-County-Commercial-Property-Tax-Defense.git
cd Dallas-County-Commercial-Property-Tax-Defense
```

---

### Install Dependencies

Suggested packages:

```txt
numpy
pandas
lightgbm
scikit-learn
matplotlib
shap
tkinter
groq
```

---

## Running the Project

---

### Run Desktop GUI

```bash
python agent_gui.py
```

---

### Run Conversational Agent

```bash
python agent_chat.py
```

---

## Important Security Note

The original Groq API key used during development has been removed from this public repository for security reasons.

If you want to run the chatbot locally, replace this section inside:

```python
agent_chat.py
```

with your own API key:

```python
GROQ_API_KEY = "your_api_key_here"
```

Never commit personal API keys to GitHub.

---

## Future Improvements

Planned enhancements include:

* comparable property retrieval engine
* GIS/location intelligence integration
* legal evidence document generation
* PDF protest defense reports
* cloud deployment for county-wide use
* ARB hearing workflow automation

---

## License

This project is intended for academic and research purposes.

For production legal or appraisal use, additional compliance review and validation would be required.

---

## Final Note

This project demonstrates how machine learning can move beyond prediction and directly support operational government decision-making.

Instead of simply predicting value, the system helps answer the real question:

### “Should the county accept this settlement—or reject it and defend the valuation?”

That is where real impact happens.
