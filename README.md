# E-Commerce-Customer-Churn-Comprehensive-EDA
Comprehensive EDA on E-Commerce Customer Churn dataset for a Data Mining course, covering churn prediction, lifetime value regression, and LTV segmentation.

## 🔍 About the Project
A structured Exploratory Data Analysis on an E-Commerce Customer Churn dataset,
following a Q1→Q20 analysis framework across 5 sections.

## 📊 Dataset
- **File:** `ecommerce_customer_churn_dataset.csv`
- **Final Shape:** 44,650 rows × 26 columns
- **3 Targets:**
  - `Churned` → Binary Classification (Churn Prediction)
  - `Lifetime_Value` → Regression (LTV Prediction)
  - `LTV_Segment` → Multiclass Classification (engineered via quantile cut into Low/Medium/High)

## 🔬 EDA Structure

### Section A — Dataset Overview & Types
- Q1: Dataset overview — shape, sample rows, column names
- Q2: Column understanding — 26 columns across 3 domains:
  demographics (Age, Gender, Country), engagement behavior
  (Login_Frequency, Session_Duration_Avg, Pages_Per_Session),
  and transaction history (Total_Purchases, Average_Order_Value, Returns_Rate)
- Q3: Type fixes — 6 float64 columns converted to Int64:
  `Age`, `Customer_Service_Calls`, `Product_Reviews_Written`,
  `Payment_Method_Diversity`, `Total_Purchases`, `Wishlist_Items`

### Section B — Data Quality
- Q4: Missing values — 15 columns had missing values concentrated
  in behavioral float columns
- Q5: Imputation strategy per column:
  - **Drop rows:** `Age`, `Customer_Service_Calls`, `Payment_Method_Diversity`
  - **Median fill (skewed):** `Days_Since_Last_Purchase` (skew=1.99),
    `Returns_Rate` (skew=5.87), `Product_Reviews_Written`, `Wishlist_Items`
  - **Mean fill (near-normal):** `Session_Duration_Avg`, `Pages_Per_Session`,
    `Discount_Usage_Rate`, `Email_Open_Rate`, `Social_Media_Engagement_Score`,
    `Mobile_App_Usage`, `Credit_Balance`
- Q6: Exact duplicates — none found
- Q7: CustomerID uniqueness — verified, each row = distinct customer
- Q8: Validity violations removed:
  - Negative `Total_Purchases`
  - `Age` outside 18–100
  - Rate columns outside 0–100
  - `Average_Order_Value` > 1000
- Q9: Categorical cleaning — strip + title/upper case applied to
  `Gender`, `Country`, `City`, `Signup_Quarter`

### Section C — Univariate EDA
- Q10: Numeric summary stats — skewness analysis per feature,
  8 columns identified as right-skewed (skew > 0.9) → log1p needed
- Q11: Histograms for all numeric features
- Q12: Categorical distributions — Gender, Country, City, Signup_Quarter

### Section D — Bivariate & Multivariate EDA
- Q13: Numeric vs Churned — t-tests per feature
- Q14: Numeric vs Lifetime_Value — Pearson correlation
- Q15: Outlier detection — boxplots per feature
- Q16: Scatter plots — feature pairs vs all 3 targets
- Q17: Categorical vs Numeric — grouped means + 3 rounds of boxplots:
  Round 1: vs Churned | Round 2: vs LTV_Segment | Round 3: vs Lifetime_Value
- Q18: Crosstab analysis — categorical vs Churned and LTV_Segment
- Q19: Full correlation heatmap — numeric + encoded categorical + all 3 targets

### Section E — Final Reporting
- Q20: Final EDA summary, top insights, risks, and modeling roadmap

## 💡 Key Insights
- **Churn Predictors:** `Cart_Abandonment_Rate` (r=0.278) and
  `Customer_Service_Calls` (r=0.289) are the strongest churn signals
- **LTV Drivers:** `Total_Purchases` (r=0.625), `Average_Order_Value` (r=0.586),
  `Session_Duration_Avg` (r=0.541) are the strongest LTV predictors
- **Demographic Features:** Gender, Country, City, Signup_Quarter show
  near-zero impact on both Churn and LTV
- **Activity Cluster:** Login_Frequency, Session_Duration_Avg, Pages_Per_Session
  are highly inter-correlated (r > 0.65) → potential dimensionality reduction
- **Class Balance:** Churn target is roughly balanced — no SMOTE needed

## ⚠️ Key Risks
- 8 right-skewed features → log1p transform required before modeling
- `Lifetime_Value` high variance → log1p transform for regression target
- `LTV_Segment` is engineered (not original) → business validation recommended
- Country and City high cardinality → Target Encoding or drop

## 🔧 Data Transformation Plan (for Modeling)
- **log1p transform:** `Membership_Years`, `Wishlist_Items`, `Total_Purchases`,
  `Days_Since_Last_Purchase`, `Returns_Rate`, `Product_Reviews_Written`,
  `Payment_Method_Diversity`, `Average_Order_Value`
- **One-Hot Encoding:** Gender, Signup_Quarter (low cardinality)
- **Target Encoding:** Country, City (high cardinality, low signal)
- **Scaling:** StandardScaler for Logistic Regression / SVM / Neural Nets —
  no scaling for tree-based models

## 🗺️ Modeling Roadmap
| Target | Task | Recommended Models |
|---|---|---|
| Churned | Binary Classification | Logistic Regression, XGBoost, Random Forest |
| Lifetime_Value | Regression | Ridge, Gradient Boosting |
| LTV_Segment | Multiclass Classification | Multiclass Logistic, Random Forest |

## 🛠️ Tools Used
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## 📂 Files
ECommerce-EDA/
│── EDA_Ecommerce.ipynb
│── ecommerce_customer_churn_dataset.csv
│── README.md
