# Factor Analysis with a Framework for Predicting Musical Residual Seats Using Machine Learning

### Abstract
The popularity of musical actors and ticket sales are closely related. However, owing to the increase in production costs caused by star casting, a vicious cycle of ticket price increases and decreased accessibility to musicals continues. To prevent this, we identify the major factors affecting ticket sales and propose a factor analysis method via a machine learning-based musical remaining seat prediction framework. The proposed framework analyzes the factors that have a significant impact on the prediction of remaining musical seats. To verify this, we conducted an experiment using nine machine learning models along with statistical analytical techniques. We analyzed the prediction results of the models using Shapley Additive exPlanations for factor analysis. The Gradient Boosting Regressor model showed the best prediction performance. We confirmed that the discount rate was the most important factor in ticket sales. Through the proposed framework, we efficiently predicted remaining seats and confirmed the most important factors in predicting remaining seats.
### 

## 📝 DOI
DOI: https://doi.org/10.9728/dcs.2025.26.3.739

## Data Availability
The datasets analyzed during the current study are not publicly available due to copyright restrictions (or privacy concerns).

## Data Format
Each CSV file should contain the following columns:

```
date, seat, cast1, cast2, cast3, cast4, weekend, day, dc, evt
```

- `date`: Performance date (format: MM/DD.T)
- `seat`: Seat sales (target variable)
- `cast1-4`: Cast information
- `weekend`: Weekend indicator (0/1)
- `day`: Day of week
- `dc`: Discount applied
- `evt`: Event indicator

## 📁 Project Structure

```
📦 musical-prediction/
├── 📄 Data Files
│   ├── bare.csv
│   ├── gentleman.csv
│   ├── hades.csv
│   ├── salieri.csv
│   └── versailles.csv
│
├── 🐍 Main Scripts
│   ├── musical_all_versions.py    # Integrated execution for all versions
│   ├── run_single_version.py      # Single version execution
│   └── compare_versions.py        # Result comparison and visualization
│
├── 🔧 Shell Scripts  
│   └── run_all_versions.sh        # Automated execution script
│
└── 📊 Results
    └── results/                   # Experiment results storage
```

## 💻 Usage Examples

### Individual Execution

```python
# Direct execution in Python
from musical_all_versions import main

# All versions
main(versions_to_run='all')

# Specific versions only
main(versions_to_run=['normal', 'smote'])

# Single version
main(versions_to_run='smote')
```
