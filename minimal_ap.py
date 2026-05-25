# minimal_app.py
import streamlit as st
import plotly.express as px
import pandas as pd

st.title("is working")

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
fig = px.bar(df, x='x', y='y', title="Test Plot")
st.plotly_chart(fig)

st.success("it works if see bar chart")