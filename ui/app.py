import streamlit as st
import requests
import pandas as pd
from sqlalchemy import create_engine
import os
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Online Shoppers Revenue Prediction", layout="wide")
st.title("Online Shoppers Revenue Prediction")

# -------------------------
# Input fields
# -------------------------
Administrative = st.number_input("Administrative", 0, 100, 0)
Administrative_Duration = st.number_input("Administrative_Duration", 0.0, 1000.0, 0.0)
Informational = st.number_input("Informational", 0, 100, 0)
Informational_Duration = st.number_input("Informational_Duration", 0.0, 1000.0, 0.0)
ProductRelated = st.number_input("ProductRelated", 0, 1000, 0)
ProductRelated_Duration = st.number_input("ProductRelated_Duration", 0.0, 2000.0, 0.0)
BounceRates = st.number_input("BounceRates", 0.0, 1.0, 0.0)
ExitRates = st.number_input("ExitRates", 0.0, 1.0, 0.0)
PageValues = st.number_input("PageValues", 0.0, 100.0, 0.0)
SpecialDay = st.number_input("SpecialDay", 0.0, 1.0, 0.0)
Month = st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"])
OperatingSystems = st.selectbox("OperatingSystems", [1,2,3,4,5])
Browser = st.selectbox("Browser", [1,2,3,4,5,6,7,8,9])
Region = st.selectbox("Region", [1,2,3,4,5,6,7,8,9])
TrafficType = st.selectbox("TrafficType", [1,2,3,4,5,6,7,8,9])
VisitorType = st.selectbox("VisitorType", ["Returning_Visitor","New_Visitor","Other"])
Weekend = st.selectbox("Weekend", ["TRUE","FALSE"])

# -------------------------
# Prediction button
# -------------------------
if st.button("Predict Revenue"):
    payload = {
        "Administrative": Administrative,
        "Administrative_Duration": Administrative_Duration,
        "Informational": Informational,
        "Informational_Duration": Informational_Duration,
        "ProductRelated": ProductRelated,
        "ProductRelated_Duration": ProductRelated_Duration,
        "BounceRates": BounceRates,
        "ExitRates": ExitRates,
        "PageValues": PageValues,
        "SpecialDay": SpecialDay,
        "Month": Month,
        "OperatingSystems": OperatingSystems,
        "Browser": Browser,
        "Region": Region,
        "TrafficType": TrafficType,
        "VisitorType": VisitorType,
        "Weekend": Weekend
    }
    
    # Call FastAPI
    response = requests.post("http://fastapi:8000/predict", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Revenue: {result['prediction']} ✅")
        st.info(f"Probability of Revenue=True: {result['probability']:.2f}")

        # -------------------------
        # Save input + prediction to DB
        # -------------------------
        try:
            db_password = os.getenv("MYSQL_PASSWORD", "Yunachan10")
            db_url = f"mysql+pymysql://sonu:{db_password}@mariadb-columnstore:3306/shoppers_db"
            engine = create_engine(db_url)

            # Convert input payload to DataFrame
            df_to_save = pd.DataFrame([payload])

            # Add Revenue column as TRUE/FALSE
            df_to_save["Revenue"] = df_to_save.apply(
                lambda row: "TRUE" if result['prediction'] == 1 else "FALSE", axis=1
            )

            # Save to a new table 'shoppers_predictions'
            df_to_save.to_sql("shoppers_predictions", con=engine, if_exists='append', index=False)
            st.success("Prediction saved to database ✅")

        except Exception as e:
            st.error(f"Failed to save prediction to DB: {e}")

    else:
        st.error("Prediction failed. Check API.")