import streamlit as st
state = st.session_state
import pandas as pd
import os
import re
import ast
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
from natsort import natsorted
import colorlover as cl
colorscale = cl.scales['6']['qual']['Set2']

def init():

    # set page configuration. Can only be set once per session and must be first st command called
    try:
        st.set_page_config(page_title='BBIP review', page_icon=':book:', layout='wide',initial_sidebar_state='collapsed')
    except st.errors.StreamlitAPIException as e:
        if "can only be called once per app" in e.__str__():
            return

    # set any custom css
    with open(os.path.abspath('./style.css')) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# @st.cache_resource
def get_data():
    # set book metadata
    df = pd.read_csv('data/metadata.csv')
    df = df[~df['music'].isnull()].copy()
    for c in ['music','violence','religion']:
        df[c] = df[c].astype(int)
    for c in ['Date','BBIP_ID']:
        df[c] = df[c].astype(int).astype(str)
    state.inventory = df

def item_head():
    head_cols = st.columns(2)
    with head_cols[0]:
        st.header(f'{state.book_md.Title}')
        st.write(f'*{state.book_md.Author} ({state.book_md.Date})*')
        st.write(f'BBIP ID: {state.book_md.BBIP_ID}')
    with head_cols[1]:

        with st.expander('About this data'):
            st.markdown("""
        This data is a result of a combination of computational methods to perform Word Sense Disambiguation (the computational process of identifying the meaning (or "sense") of a word in the context of the surrounding text). Initial processing was done via BookNLP (https://github.com/booknlp/booknlp), a natural language processing pipeline used to identify a variety of textual features in novel-length texts. Definitions and word usage were based on the Wordnet model (https://wordnet.princeton.edu/), with definitions and vocabulary expanded by the 2022 version of the Open English Wordnet (https://en-word.net/). A transformer based BERT model, GlossBERT (https://github.com/HSLCY/GlossBERT) (trained on the Wordnet corpus), was used to evaluate each noun and verb in each text, with the results compared to a pre-identified list of over 3,000 topic synsets. \n\nNote that computational word sense disambiguation is an imprecise process, due to the complex nature of human language. This data may have some ommissions and/or inaccurate matches.""")

def get_description():
    st.write('This is a list of all of the instances of a given topic within the text, as identified through the processes described in "About this data" (above). Note that if a word is identified twice in a given sentence, it will appear twice in the results.')
    st.markdown("""**Definitions:**

* **synset:** a group of words that share the same meaning ("sense") in a specific context. Example: the synset for the sense "bank as a financial institution" includes the words "bank," "depository," and "savings bank." These take the form of "word.part_of_speech.definition_num", e.g. "bank.n.01"
* **lemma:** the base or dictionary form of a word, which represents all its inflected forms. Example: the lemma "run" represents the words "run", "runs," "ran," and "running. Lemmas are used to identify synsets in this context.
* **word:** the word as it appears in the text""")

    st.markdown("""**Sample result sorted by synset**:

    Format:
        synset - synset definition - (number of occurrences in the text) uses
            * [a list of each occurrence follows]
            * word (part_of_speech) -- sentence in which the word appears
    Example:
        abuse.v.03 - use foul or abusive language towards - (1) uses
            * shouted (verb) -- Cook stood up and shouted back at her , ‘ Watch your tongue , you foul beast !
            """)

    st.markdown("""**Sample result sorted by word**:

    Format:
        lemma - (number of occurrences in the text) uses
            * [a list of each occurrence follows]
            * word (part of speech) -- sentence in which the word appears
                * synset -- synset definition

    Example:
        chant - (2) uses
            * chanted (verb) -- The pickets formed a circle in front of the Back - to - the - Southland office and chanted as they marched , “ Go , white man , go while you can ....
                * chant.v.01 -- recite with musical intonation; recite as a chant or a psalm
            * chanting (verb) -- They were chanting , ‘ We want O’Malley ....
                * chant.v.01 -- recite with musical intonation; recite as a chant or a psalm
            """)

def set_colors(df):

    tops = natsorted(df[~df.topic.isnull()].topic.unique())
    state.colorlist = {topic: colorscale[i] for i, topic in enumerate(tops)}


# @st.cache_resource
def display_table():

    sort_top_cols = st.columns(4)
    with sort_top_cols[0]:
        state.sort_top = st.selectbox('Sort titles by', state.inventory.columns, key='st')
    with sort_top_cols[1]:
        state.order_top = st.radio('Order results', ['Ascending', 'Descending'], key='ot', horizontal=True)

    if 'sort_top' not in state:
        state.sort_top = 'BBIP_ID'
    if 'order_top' not in state:
        state.order_top = 'Ascending'
    order_top = False if state.order_top == 'Descending' else True

    link = "text-decoration: none; color: inherit;"
    td = "padding: 0 1em; border: 1px solid #BCBCBC"
    content = f"""<div id="inv"><table style='border-collapse: collapse;'><thead><tr><th style='{td}'>Title</th><th style='{td}'>Author</th><th style='{td}'>Publication date</th><th style='{td}'>BBIP ID</th>
    <th style='{td}'>Music</th><th style='{td}'>Violence</th><th style='{td}'>Religion</th></row></thead><tbody>"""

    state.inventory = state.inventory.sort_values(by=state.sort_top, ascending=order_top)

    for i,r in state.inventory.iterrows():

        row = f"""<tr>
                <td style='{td}'><a href='#' style='{link}' id='{r.BBIP_ID}'>{r.Title}</a></td>
                <td style='{td}'>{r.Author}</td>
                <td style='{td}'>{r.Date}</td>
                <td style='{td}'><a href='#' style='{link}' id='{r.BBIP_ID}'>{r.BBIP_ID}</a></td>
                <td style='{td}'>{r.music}</td>
                <td style='{td}'>{r.violence}</td>
                <td style='{td}'>{r.religion}</td></tr>"""

        content += row
    content = content + '</tbody></table></div>'
    return content

# @st.cache_resource
def get_wc(weighted_txt):

    # generate a word cloud image:
    w,h = 800,400
    wordcloud = WordCloud(background_color="white", colormap='twilight',
        width=w, height=h, collocations=False, max_words=1000, margin=5).fit_words(weighted_txt)

    # Display the generated image:
    plt.figure()
    wc = plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return wc

# @st.cache_resource
def topic_graph(t, df):

    # group and plot together
    start = df.sentence_ID.min()
    end = df.sentence_ID.max()
    num_groups = 10
    total_range = end - start + 1
    group_size = total_range // num_groups

    # Create the bins
    bins = [start + i * group_size for i in range(num_groups + 1)]
    bins[-1] = end + 1  # To include the end value in the last bin

    if t in df.topic.unique():
        tdf = df[df.topic==t].copy()
    else:
        tdf = df[~df.topic.isnull()].copy()
    tdf['binned'] = pd.cut(tdf['sentence_ID'], bins=bins, right=False)

    fig = go.Figure()

    # Count the occurrences in each bin and format the output for better readability
    for term in tdf.topic.unique():
        counts = tdf[tdf.topic==term]['binned'].value_counts().sort_index()
        counts = counts.rename(lambda x: f"{x.left} - {x.right - 1}", axis='index')
        x = counts.index
        y = counts.values

        fig.add_trace(go.Scatter(x=x, y=y,
                            name=f'{term} ({len(tdf[tdf.topic==term]):,} total uses)',
                            # text=kwsearch_abs,
                            mode='lines', connectgaps= True, line_shape='spline', line=dict(color=state.colorlist[term])))
    fig.update_layout(xaxis_title='Occurence in the text (10 parts)',
                        yaxis_title='Number of occurrences')

    return fig

def book_sum(t, df):

    bk_sum_1, bk_sum_2 = st.columns(2)

    def top_cts(tdf):
        top_all = len(tdf)
        top_unique = len(tdf.sentence_ID.unique())
        p = (top_unique / state.book_md.total_sents) * 100
        return top_all, top_unique, p

    with bk_sum_1:

        if t in df.topic.unique():
            with st.container(border=True):
                tdf = df[df.topic==t]
                top_all, top_unique, p = top_cts(tdf)
                st.markdown(f'*Topic appears **{top_all:,}** times in **{top_unique:,}** sentences, or **{p:.1f}%** of the text.*')

        else:
            with st.container(border=True):
                st.write(f'Word count: **{state.book_md.wordcount:,}**')
                all_sents = state.book_md.total_sents
                st.write(f'Sentence count: **{all_sents:,}**')

                tdf = df[~df.topic.isnull()]
                top_all, top_unique, p = top_cts(tdf)
                st.markdown(f'All topics: **{top_all:,}** occurrences in **{top_unique:,}** sentences (**{p:.1f}%** of the text)')
                top_list = ''
                for t in tdf.topic.unique():
                    top_all, top_unique, p = top_cts(tdf[tdf.topic==t])
                    top_list += f'* ***{t}***: **{top_all:,}** occurrences in **{top_unique:,}** sentences (**{p:.1f}%** of the text)\n'
                st.write(top_list)


    if t == '*':

        with bk_sum_2:

            data = {}
            for t in tdf.topic.unique():
                data[t] = len(tdf[tdf.topic==t].sentence_ID.unique())
            # data['Full text'] = all_sents

            topics = list(data.keys())
            counts = list(data.values())

            barcolors = [state.colorlist.get(top, colorscale[-1]) for top in topics]
            fig = go.Figure(data=[go.Bar(x=topics, y=counts, marker_color=barcolors)])
            fig.update_layout(
                title='Topic count',
                xaxis_title='Topic',
                yaxis_title='Number of sentences',
            )
            st.plotly_chart(fig)

def topic_viz(t, df):

    if t in df.topic.unique():
        tdf = df[df.topic==t]
        st.markdown(f'### Review the identified words for this topic: {t.title()}')

    else:
        tdf = df[~df.topic.isnull()]
        st.markdown('### Review all topic words')
    td_cols = st.columns((1,2,2))

    with td_cols[0]:
        st.write('*10 most frequent terms*')
        # st.write(pd.DataFrame(tdf.lemma.value_counts()[:10]))
        table = "| term | count |\n"
        table += "|--------|-------|\n"
        for k,v in tdf.lemma.value_counts()[:10].items():
            table += f'| {k} | {v} |\n'

        st.markdown(table, unsafe_allow_html=True)

    with td_cols[1]:
        st.write('*Wordcloud of terms by frequency*')
        weighted_txt = tdf.lemma.str.lower().value_counts().to_dict()
        wc = get_wc(weighted_txt)
        st.write(wc.figure)

    with td_cols[2]:
        st.markdown('*Appearance of topic within the book*')
        st.write(topic_graph(t, tdf))

def list_synset(g):
    lst = ''
    for i,r in g.iterrows():
        context = re.sub(r.word, f'**{r.word}**', r.context, flags=re.IGNORECASE)
        lst += f"* **{r.word.lower()}** ({r.POS_tag.lower()}) -- {context}\n"
    st.write(lst)
    if state.expand_all == 'Expanded':
        st.write('<hr/>', unsafe_allow_html=True)
def list_word(g):
    lst = ''
    for i,r in g.iterrows():
        context = re.sub(r.word, f'**{r.word.lower()}**', r.context, flags=re.IGNORECASE)
        lst += f"""* **{r.word.lower()}** ({r.POS_tag.lower()}) -- {context}
    * ***{r.synset}*** -- {r.definition}\n"""
    st.write(lst)
    if state.expand_all == 'Expanded':
        st.write('<hr/>', unsafe_allow_html=True)

def topic_display(t, df):

    tdf = df[df.topic==t][['word','lemma','POS_tag','synset','definition','context','lemmas']]

    st.markdown('### Review all topic matches')

    sort_all_cols = st.columns(4)
    sort_opts = ['synset','word']
    with sort_all_cols[0]:
        state.sort_all = st.selectbox('Sort matches by', sort_opts, key=f'sa_{t}')
    with sort_all_cols[1]:
        state.order_all = st.selectbox('Order results', ['Alphabetically A-Z', 'Alphabetically Z-A', 'Most frequent',
        'Least frequent'], key=f'oa_{t}')
    with sort_all_cols[2]:
        state.expand_all = st.radio('Show results', ['Expanded', 'Collapsed'], key=f'ea_{t}', horizontal=True)

    if 'sort_all' not in state:
        state.sort_all = 'synset'
    if 'order_all' not in state:
        state.order_all = 'Alphabetically A-Z'
    if 'expand_all' not in state:
        state.expand_all = 'Expanded'

    order_all = True if state.order_all in ['Alphabetically Z-A','Most frequent'] else False
    sort_all = 'lemma' if state.sort_all == 'word' else 'synset'

    with st.expander('What do these results mean?'):
        get_description()

    grouped = tdf.groupby(sort_all)
    if 'Alphabetically' in state.order_all:
        sorted_groups = sorted(grouped.groups.keys(), key=str.lower, reverse=order_all)
    else:
        sorted_groups = sorted(grouped.groups.keys(), key=lambda x: (len(grouped.get_group(x)), x.lower()), reverse=order_all)
    for n in sorted_groups:
        g = grouped.get_group(n)
        if sort_all == 'synset':
            if state.expand_all == 'Expanded':
                st.write(f"***{n}*** - {g['definition'].iloc[0]} - ***({len(g)}) uses***")
                list_synset(g)
            else:
                with st.expander(f"***{n}*** - {g['definition'].iloc[0]} - ***({len(g)}) uses***"):
                    list_synset(g)

        else:
            if state.expand_all == 'Expanded':
                st.write(f"***{n}*** - ***({len(g)}) uses***")
                list_word(g)
            else:
                with st.expander(f"***{n}*** - ***({len(g)}) uses***"):
                    list_word(g)

                # with st.expander(f"***{n}*** - {g['definition'].iloc[0]} - ***({len(g)}) uses***"):
                #     tbl = f"""| Word | Part of speech | Sentence |\n| -- | -- | -- |\n"""
                #     for i,r in g.iterrows():
                #         tbl += f"| {r.word.lower()} | {r.POS_tag.lower()} | {r.context} |\n"
                #     st.write(tbl, unsafe_allow_html=True)

            # else:
            #     with st.expander(f"***{n}*** - ***({len(g)}) uses***"):
            #         tbl = f"""| Word | Part of speech | Sentence |\n| -- | -- | -- |\n"""
            #         for i,r in g.iterrows():
            #             tbl += f"| {r.word.lower()} | {r.POS_tag.lower()} | {r.context} |\n"
            #         st.write(tbl, unsafe_allow_html=True)
