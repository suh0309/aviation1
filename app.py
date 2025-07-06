
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
@st.cache
def load_data():
    return pd.read_csv('passenger_journey.csv')

data = load_data()

st.sidebar.title("Analytics Dashboard")
module = st.sidebar.radio("Select Module", ["Pre-Flight", "In-Flight", "Post-Flight"])

if module == "Pre-Flight":
    st.header("Pre-Flight Analysis & Models")

    # Visualization
    st.subheader("Wait Time Distribution")
    st.bar_chart(data['wait_time'].value_counts().sort_index())

    # Linear Regression: Predict wait_time
    st.subheader("Linear Regression: Predicting Wait Time")
    X = data[['map_open', 'fast_track']]
    y = data['wait_time']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    preds = lr.predict(X_test)
    st.write(f"R² score: {r2_score(y_test, preds):.2f}")
    st.write("Coefficients:", dict(zip(X.columns, lr.coef_.round(2))))

    # Association Rule Mining
    st.subheader("Association Rules on Behaviors")
    pf = data[['map_open', 'coffee_purchase', 'fast_track']].astype(bool)
    freq = apriori(pf, min_support=0.1, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.2)
    st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

elif module == "In-Flight":
    st.header("In-Flight Analysis & Models")

    # Visualization
    st.subheader("Ancillary Spend vs Flight Duration")
    st.line_chart(data.groupby('flight_duration')['ancillary_spend'].mean())

    # Classification: Predict upgrade_purchase
    st.subheader("Classification: Predicting Upgrade Purchase")
    X = data[['mood_score']]
    y = data['upgrade_purchase']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = LogisticRegression(max_iter=200).fit(X_train, y_train)
    preds = clf.predict(X_test)
    st.write(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
    st.write("Coefficients:", dict(zip(X.columns, clf.coef_[0].round(2))))

    # Clustering: IFE engagement clusters
    st.subheader("Clustering: IFE Engagement Segments")
    Xc = data[['mood_score', 'ancillary_spend']]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(Xc)
    data['cluster'] = kmeans.labels_
    st.write(data['cluster'].value_counts())

else:
    st.header("Post-Flight Analysis")

    # Visualization
    st.subheader("Baggage Wait Time Distribution")
    st.bar_chart(data['baggage_wait'].value_counts().sort_index())

    st.subheader("Feedback Score Distribution")
    st.bar_chart(data['feedback_score'].value_counts().sort_index())

    st.subheader("Transfer Booking Rate")
    st.bar_chart(data['transfer_booked'].value_counts(normalize=True))

st.sidebar.markdown("Upload new data:")
uploaded_file = st.sidebar.file_uploader("", type=["csv"])
if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data uploaded!")
    st.sidebar.download_button("Download CSV", new_data.to_csv(index=False), "processed.csv")
