import streamlit as st
state = st.session_state
import pandas as pd
import ast

# initialize
from scripts import tools
tools.init()

if 'bbipid' not in state:
    st.switch_page('home.py')

if st.button('Return to main page'):
    st.switch_page('home.py')

state.book_md = state.inventory[state.inventory.BBIP_ID==state.bbipid].iloc[0]
tools.item_head()

df = pd.read_csv(f'data/{state.bbipid}')
df['lemmas'] = df['lemmas'].apply(ast.literal_eval)
tools.set_colors(df)

md_cols = st.columns(2)

mus = len(df[df.topic=='music'])
vio = len(df[df.topic=='violence'])
rel = len(df[df.topic=='religion'])

tdf = df[~df.topic.isnull()]
bt1, bt2, bt3, bt4 = st.tabs(['Overview', 'Music', 'Violence', 'Religion'])

with bt1:
    tools.book_sum('*', df)
    tools.topic_viz('*', df)

with bt2:
    tools.book_sum('music', df)
    tools.topic_viz('music', df)
    tools.topic_display('music', df)

with bt3:
    tools.book_sum('violence', df)
    tools.topic_viz('violence', df)
    tools.topic_display('violence', df)

with bt4:
    tools.book_sum('religion', df)
    tools.topic_viz('religion', df)
    tools.topic_display('religion', df)
