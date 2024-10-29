import streamlit as st
state = st.session_state
from st_click_detector import click_detector as cd
import pandas as pd

# initialize
from scripts import tools
tools.init()
tools.get_data()

st.header('HBW - 40 books review')

st.write('*Click any title or BBIP ID to see more details.*')
content = tools.display_table()
clicked = cd(content)

if clicked:
    state.bbipid = clicked
    st.switch_page("itemview.py")
