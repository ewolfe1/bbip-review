import streamlit as st

pg = st.navigation([st.Page("home.py"), st.Page("itemview.py")], position="hidden")
pg.run()
