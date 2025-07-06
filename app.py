import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

@st.cache
def load_data():
    return pd.read_csv('passenger_journey.csv')

data = load_data()

st.sidebar.title("Interactive Analytics Dashboard")
module = st.sidebar.radio("Select Module", ["Pre-Flight", "In-Flight", "Post-Flight"])

if module == "Pre-Flight":
    st.header("Pre-Flight Analysis & Models")

    # Variable selection
    numeric_cols = ['wait_time', 'fast_track', 'map_open', 'coffee_purchase']
    x_var = st.selectbox("X variable", numeric_cols, index=2)
    y_var = st.selectbox("Y variable", numeric_cols, index=0)

    # Scatter plot
    fig = px.scatter(
        data,
        x=x_var,
        y=y_var,
        title=f"{y_var.replace('_',' ').title()} vs {x_var.replace('_',' ').title()}",
        labels={x_var: x_var.replace('_',' ').title(), y_var: y_var.replace('_',' ').title()}
    )
    st.plotly_chart(fig, use_container_width=True)

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
    st.subheader("Association Rules on Pre-Flight Behaviors")
    pf = data[['map_open', 'coffee_purchase', 'fast_track']].astype(bool)
    freq = apriori(pf, min_support=0.1, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.2)
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

elif module == "In-Flight":
    st.header("In-Flight Analysis & Models")

    # Variable selection
    numeric_cols = ['flight_duration', 'ancillary_spend', 'mood_score', 'upgrade_purchase']
    x_var = st.selectbox("X variable", numeric_cols, index=0)
    y_var = st.selectbox("Y variable", numeric_cols, index=1)

    # Line chart
    pivot = data.groupby(x_var)[y_var].mean().reset_index()
    fig = px.line(
        pivot,
        x=x_var,
        y=y_var,
        title=f"Mean {y_var.replace('_',' ').title()} by {x_var.replace('_',' ').title()}",
        labels={x_var: x_var.replace('_',' ').title(), y_var: y_var.replace('_',' ').title()}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Classification: Predict upgrade_purchase
    st.subheader("Classification: Predicting Upgrade Purchase")
    X = data[['mood_score']]
    y = data['upgrade_purchase']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = LogisticRegression(max_iter=200).fit(X_train, y_train)
    preds = clf.predict(X_test)
    st.write(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
    st.write("Coefficient:", clf.coef_[0][0].round(2))

    # Clustering: IFE engagement segments
    st.subheader("Clustering: IFE Engagement Segments")
    Xc = data[['mood_score', 'ancillary_spend']]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(Xc)
    data['cluster'] = kmeans.labels_
    df_clusters = data['cluster'].value_counts().rename_axis('cluster').reset_index(name='count')
    fig2 = px.bar(
        df_clusters,
        x='cluster',
        y='count',
        title="IFE Engagement Cluster Counts",
        labels={'cluster': 'Cluster', 'count': 'Count'}
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.header("Post-Flight Analysis")

    # Variable selection
    numeric_cols = ['baggage_wait', 'feedback_score', 'transfer_booked', 'loyalty_triggered']
    x_var = st.selectbox("Select variable for distribution", numeric_cols, index=0)

    # Build distribution dataframe
    df_dist = data[x_var].value_counts().reset_index()
    df_dist.columns = [x_var, 'count']

    fig3 = px.bar(
        df_dist,
        x=x_var,
        y='count',
        title=f"Distribution of {x_var.replace('_',' ').title()}",
        labels={x_var: x_var.replace('_',' ').title(), 'count': 'Count'}
    )
    st.plotly_chart(fig3, use_container_width=True)

# Sidebar file upload
st.sidebar.markdown("### Upload New Data")
uploaded_file = st.sidebar.file_uploader("", type=["csv"])
if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data uploaded!")
    st.sidebar.download_button("Download CSV", new_data.to_csv(index=False), "processed.csv")
