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

def get_stats(df):

    # get some relative freqs
    for t in ['music','violence','religion']:

        nm = t.capitalize()
        t = f'{t}_words'

        # Standard z-score for raw topic counts
        mean = df[t].mean()
        std = df[t].std()
        df[f'{nm} (Z raw)'] = (df[t] - mean) / std

        # Frequency per 1,000 words
        df[f'{nm} (% per 1000)'] = (df[t] / df['total_words']) * 1000

        # Z-score for relative frequency
        rel = df[t] / df['total_words']
        rel_mean = rel.mean()
        rel_std = rel.std()
        df[f'{nm} (Z rel)'] = (rel - rel_mean) / rel_std

    return df

# @st.cache_resource
def get_data():
    # set book metadata
    df = pd.read_csv('data/metadata.csv')
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], inplace=True, axis=1)
    # df = df[~df['music'].isnull()].copy()
    for c in df.columns:
        if c in ['Date','BBIP_ID']:
            df[c] = df[c].astype(int).astype(str)
        elif c == 'alt_BBIP_IDs':
            continue
        else:
            try:
                df[c] = df[c].astype(int)
            except Exception as e:
                continue

    df['alt_BBIP_IDs'] = df['alt_BBIP_IDs'].apply(lambda x: str(x).split('.')[0] if pd.notna(x) and not isinstance(x, str) else x)

    # calculate frequencies
    df = get_stats(df)

    col_map = {'total_words':'Total words', 'total_sents':'Total sentences','topic_words':'Topic words (all)','topic_sents':'Topic sentences (all)','religion_words':'Religion (WC)','religion_sents':'Religion (SC)','violence_words':'Violence (WC)','violence_sents':'Violence (SC)','music_words':'Music (WC)','music_sents':'Music (SC)','alt_BBIP_IDs':'Alt. BBIP IDs','Coll_40_books':'40 books'}
    df.rename(columns=col_map, inplace=True)

    state.inventory_full = df
    state.inventory = df
    state.default_cols = ['Title','Author','Date','BBIP_ID','Alt. BBIP IDs','Total words']


def set_colors(df):

    tops = natsorted(df[~df.topic.isnull()].topic.unique())
    state.colorlist = {topic: colorscale[i] for i, topic in enumerate(tops)}

def filter_inv():

    # first, set the base inventory
    if 'display40' not in state:
        state.display40 = 'All titles'
    if state.display40 == 'All titles':
        state.inventory = state.inventory_full
    else:
        state.inventory = state.inventory_full[state.inventory_full['40 books']=='Y']

    # search filter
    if 'search' not in state:
        state.search = ''
    state.inventory = state.inventory[state.inventory.Title.str.contains(state.search, na=False, case=False) | state.inventory.Author.str.contains(state.search, na=False, case=False) | state.inventory.BBIP_ID.str.contains(state.search, na=False, case=False)]

    # topics
    if 'topics' not in state:
        state.topics = 'All (Z rel)'
    # error handling hack
    try:
        if 'All' in state.topics:
            pass
    except (TypeError, KeyError) as e:
        state.topics = 'All (Z rel)'

    if 'All' in state.topics:
        stat = re.findall(r'\((.*?)\)', state.topics)[0]
        state.display_cols = state.default_cols + [f'Music ({stat})',f'Religion ({stat})',f'Violence ({stat})']
    else:
        state.display_cols = state.default_cols + [c for c in state.inventory.columns if state.topics in c]
    state.inventory = state.inventory[state.display_cols]

# @st.cache_resource
def display_table():

    filter_cols = st.columns(3)

    with filter_cols[0]:
        state.topics = st.pills('*Topic stats to display*', ['Music','Violence','Religion','All (WC)','All (Z raw)','All (% per 1000)','All (Z rel)'], selection_mode='single',default=['All (Z rel)'], help=stats_desc)

    with filter_cols[1]:
        state.search = st.text_input('*Search by title, author, or BBIP ID*')

    with filter_cols[2]:
        state.display40 = st.pills('*Texts to display*', ['All titles', '40 books'], default=['All titles'], key='dt', selection_mode='single')

    stats_desc = """**WC** - The raw count of words matching the topic within the text.\n
**Z raw** - The z-score of the raw word count, which standardizes the raw topic count by comparing it to the mean and standard deviation of all raw counts across the dataset.\n
**% per 1,000** - The percentage of topic occurrences normalized to 1,000 words, calculated as (topic count/total word count) × 1,000.\n
**Z rel** - The z-score of the relative word count, which standardizes the proportion of topic words to total words by comparing it to the mean and standard deviation of all relative word counts across the dataset.\n"""

    filter_inv()

    st.write(f'***{len(state.inventory)} titles displayed***')

    clicked = st.dataframe(state.inventory, use_container_width=True, hide_index=True, key="bk_display", selection_mode="single-row", on_select="rerun")

    if clicked:
        return clicked

def item_head():
    head_cols = st.columns(2)
    with head_cols[0]:
        st.header(f'{state.book_md.Title}')
        st.write(f'*{state.book_md.Author} ({state.book_md.Date})*')
        st.write(f'**BBIP ID**: {state.book_md.BBIP_ID}')
        if not pd.isnull(state.book_md['Alt. BBIP IDs']):
            st.write(f'\t*Alternate BBIP IDs:* {state.book_md["Alt. BBIP IDs"]}')
        if not pd.isnull(state.book_md['40 books']):
            st.write(':books: *This book is one the 40 Books for 40 Years collection.* :books:')
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

def book_stats(t):

    cts = pd.DataFrame(state.book_md[[c for c in state.book_md.index if t.lower() in c.lower()]]).T
    cts.columns = ['Word count (raw)','Sentence count','Z-score (raw)', 'Frequency per 1,000 words','Z-score (relative)']
    cts = cts.T
    cts.columns = ['value']

    summ = ""
    for i,r in cts.iterrows():
        if isinstance(r.value, float):
            summ += f'* **{i}:** {r.value:.3f}\n'
        else:
            summ += f'* **{i}:** {r.value}\n'

    st.write(summ)

def book_sum(t, df):

    bk_sum_1, bk_sum_2 = st.columns(2)

    def top_cts(tdf):
        top_all = len(tdf)
        top_unique = len(tdf.sentence_ID.unique()) + 1
        p = (top_unique / state.book_md['Total sentences']) * 100
        return top_all, top_unique, p



    if t in df.topic.unique():
        with bk_sum_1:
            with st.container(border=True):
                tdf = df[df.topic==t]
                top_all, top_unique, p = top_cts(tdf)
                st.markdown(f'*Topic appears **{top_all:,}** times in **{top_unique:,}** sentences, or **{p:.1f}%** of the text.*')

                book_stats(t)
        with bk_sum_2:
            with st.expander('About the statistics'):
                st.markdown("""
            **Word count (raw)** - The raw count of words matching the topic within the text.\n
            **Sentence count** - The number of sentences within the text containing the topic.\n
            **Z-score (raw)** - The z-score of the raw word count, which standardizes the raw topic count by comparing it to the mean and standard deviation of all raw counts across the dataset.\n
            **Frequency per 1,000 words** - The percentage of topic occurrences normalized to 1,000 words, calculated as (topic count/total word count) × 1,000.\n
            **Z-score (relative)** - The z-score of the relative word count, which standardizes the proportion of topic words to total words by comparing it to the mean and standard deviation of all relative word counts across the dataset. This should be the best metric to compare texts to each other.\n""")

    else:
        with bk_sum_1:
            with st.container(border=True):
                st.write(f'Word count: **{state.book_md["Total words"]:,}**')
                all_sents = state.book_md['Total sentences']
                st.write(f'Sentence count: **{all_sents:,}**')

                tdf = df[~df.topic.isnull()]
                top_all, top_unique, p = top_cts(tdf)
                st.markdown(f'All topics: **{top_all:,}** occurrences in **{top_unique:,}** sentences (**{p:.1f}%** of the text)')
                top_list = ''
                for t in tdf.topic.unique():
                    top_all, top_unique, p = top_cts(tdf[tdf.topic==t])
                    top_list += f'* ***{t}***: **{top_all:,}** occurrences in **{top_unique:,}** sentences (**{p:.1f}%** of the text)\n'
                st.write(top_list)



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
