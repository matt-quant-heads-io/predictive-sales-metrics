import streamlit as st

tickers = st.text_input("Enter a list of tickers (separated by commas)", "(e.g. AAPL, GOOGL)")
st.write("The list of tickers is", tickers)

st.button("Reset", type="primary")
if st.button("Run"):
    st.write("Why hello there")
else:
    st.write("Goodbye")