import streamlit as st
state = st.session_state
from st_click_detector import click_detector as cd
import pandas as pd

# initialize
from scripts import tools
tools.init()
tools.get_data()

st.header('History of Black Writing - text review')

st.write('*Select any row to see a detailed topic analysis. Select any column header to sort the data.*')

clicked = tools.display_table()

if len(clicked.selection['rows']) > 0:
    state.bbipid = state.inventory.iloc[clicked.selection['rows'][0]]['BBIP_ID']
    st.switch_page("itemview.py")
