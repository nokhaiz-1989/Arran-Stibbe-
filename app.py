import streamlit as st

st.set_page_config(
    page_title="Advanced Eco-CDA Analyzer",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import re
from collections import Counter

import anthropic
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from wordcloud import WordCloud

# ─────────────────────────────────────────────────────────────────────────────
# NLTK
# ─────────────────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.corpus import stopwords
from nltk.util import ngrams

# ─────────────────────────────────────────────────────────────────────────────
# spaCy
# ─────────────────────────────────────────────────────────────────────────────
import spacy


@st.cache_resource(show_spinner="Loading spaCy model...")
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")

    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


nlp = load_nlp()

# ─────────────────────────────────────────────────────────────────────────────
# Stopwords
# ─────────────────────────────────────────────────────────────────────────────
STOP = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────────────────────
# Dictionaries
# ─────────────────────────────────────────────────────────────────────────────
METAPHOR_WORDS = [
    "battle",
    "fight",
    "combat",
    "enemy",
    "tsunami",
]

EVALUATION_WORDS = [
    "devastating",
    "catastrophic",
    "severe",
    "tragic",
    "dangerous",
    "massive",
    "horrific",
    "destructive",
]

IDENTITY_WORDS = [
    "government",
    "victims",
    "families",
    "citizens",
    "farmers",
    "authorities",
    "communities",
    "people",
    "minister",
    "official",
    "community",
    "citizen",
]

IDEOLOGY_WORDS = [
    "development",
    "progress",
    "responsibility",
    "policy",
    "national",
    "economic",
    "climate",
    "sustainability",
]

ERASURE_PATTERNS = [
    "were displaced",
    "was destroyed",
    "were affected",
    "was damaged",
    "lost their homes",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(text: str):

    text = re.sub(r"http\S+", "", str(text).lower())
    text = re.sub(r"[^a-z\s]", " ", text)

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and token.text not in STOP
        and not token.is_stop
        and len(token.text) > 1
    ]

    return tokens


def classify(sentence: str):

    doc = nlp(sentence.lower())

    words = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha
    ]

    s = " ".join(words)

    cats = []

    # METAPHOR
    metaphor_patterns = [
        "battle against",
        "fight against",
        "war on",
        "wave of",
    ]

    if any(p in s for p in metaphor_patterns):
        cats.append("Metaphor")

    elif any(w in words for w in METAPHOR_WORDS):

        climate_context = [
            "climate",
            "flood",
            "disaster",
            "environment",
            "crisis",
        ]

        if any(c in words for c in climate_context):
            cats.append("Metaphor")

    # EVALUATION
    if any(w in words for w in EVALUATION_WORDS):
        cats.append("Evaluation")

    # IDENTITY
    if any(w in words for w in IDENTITY_WORDS):
        cats.append("Identity")

    # IDEOLOGY
    if any(w in words for w in IDEOLOGY_WORDS):
        cats.append("Ideology")

    # ERASURE
    original_sentence = sentence.lower()

    if any(p in original_sentence for p in ERASURE_PATTERNS):
        cats.append("Erasure")

    return cats


def word_freq(tokens):
    return Counter(tokens)


def concordance(text, keyword, width=60):

    out = []

    for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):

        start = max(match.start() - width, 0)
        end = min(match.end() + width, len(text))

        left = text[start:match.start()]
        kw = text[match.start():match.end()]
        right = text[match.end():end]

        out.append([left, kw, right])

    return out


def count_words(text):
    return len(text.split())


def get_sentiment(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        return "Positive"

    elif polarity < -0.1:
        return "Negative"

    return "Neutral"


def generate_ngrams(tokens, n=2, top_k=15):

    grams = ngrams(tokens, n)

    gram_freq = Counter(grams)

    return pd.DataFrame(
        [
            (" ".join(k), v)
            for k, v in gram_freq.most_common(top_k)
        ],
        columns=["Phrase", "Frequency"]
    )


def extract_entities(text):

    doc = nlp(text)

    entities = [
        (ent.text, ent.label_)
        for ent in doc.ents
    ]

    entity_df = pd.DataFrame(
        entities,
        columns=["Entity", "Type"]
    )

    if entity_df.empty:
        return entity_df

    return (
        entity_df
        .value_counts()
        .reset_index(name="Frequency")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stories We Live By — Corpus Sample Helper
# ─────────────────────────────────────────────────────────────────────────────

def prepare_corpus_sample(text, max_chars=4000):
    """
    Takes a balanced sample from beginning, middle and end
    of the corpus so large texts fit within the API context.
    """
    if len(text) <= max_chars:
        return text

    third = max_chars // 3

    sample = (
        text[:third]
        + "\n...\n"
        + text[len(text) // 2: len(text) // 2 + third]
        + "\n...\n"
        + text[-third:]
    )

    return sample


# ─────────────────────────────────────────────────────────────────────────────
# Stories We Live By — Claude API Function
# ─────────────────────────────────────────────────────────────────────────────

def generate_stibbe_stories(corpus_text: str):
    """
    Calls the Claude API once per Stibbe story type and returns
    a dictionary of {story_type: short_story_text}.
    """

    client = anthropic.Anthropic(
        api_key=st.secrets["ANTHROPIC_API_KEY"]
    )

    sample = prepare_corpus_sample(corpus_text)

    stibbe_prompts = {

        "Framing Story": f"""
You are an ecolinguist applying Arran Stibbe's Stories We Live By 
framework from his 2015 book Ecolinguistics: Language, Ecology and 
the Stories We Live By.

Analyse the corpus below and write a short story of 3 to 5 sentences 
that reveals the FRAMING STORY — how the topic is conceptually bounded, 
what assumptions are built into the way the subject is presented, 
and what kind of world the text takes for granted.

Ground every sentence in specific evidence from the corpus.
Write in plain English. Do not use bullet points.

Corpus:
{sample}
""",

        "Salience Story": f"""
You are an ecolinguist applying Arran Stibbe's Stories We Live By 
framework from his 2015 book Ecolinguistics: Language, Ecology and 
the Stories We Live By.

Analyse the corpus below and write a short story of 3 to 5 sentences 
that reveals the SALIENCE STORY — what is foregrounded and made 
highly visible in the text, and what is pushed into the background 
or mentioned only briefly.

Ground every sentence in specific evidence from the corpus.
Write in plain English. Do not use bullet points.

Corpus:
{sample}
""",

        "Metaphor Story": f"""
You are an ecolinguist applying Arran Stibbe's Stories We Live By 
framework from his 2015 book Ecolinguistics: Language, Ecology and 
the Stories We Live By.

Analyse the corpus below and write a short story of 3 to 5 sentences 
that reveals the METAPHOR STORY — the conceptual metaphors that 
structure how the topic is perceived, what they reveal about 
underlying attitudes, and what they conceal.

Ground every sentence in specific evidence from the corpus.
Write in plain English. Do not use bullet points.

Corpus:
{sample}
""",

        "Identity Story": f"""
You are an ecolinguist applying Arran Stibbe's Stories We Live By 
framework from his 2015 book Ecolinguistics: Language, Ecology and 
the Stories We Live By.

Analyse the corpus below and write a short story of 3 to 5 sentences 
that reveals the IDENTITY STORY — how different groups, communities, 
institutions and individuals are constructed and positioned in the text, 
who is given a voice and who is spoken about but never heard.

Ground every sentence in specific evidence from the corpus.
Write in plain English. Do not use bullet points.

Corpus:
{sample}
""",

        "Erasure Story": f"""
You are an ecolinguist applying Arran Stibbe's Stories We Live By 
framework from his 2015 book Ecolinguistics: Language, Ecology and 
the Stories We Live By.

Analyse the corpus below and write a short story of 3 to 5 sentences 
that reveals the ERASURE STORY — what voices, perspectives, living 
beings or realities are systematically absent or written out of 
the text entirely, and what effect this absence creates.

Ground every sentence in specific evidence from the corpus.
Write in plain English. Do not use bullet points.

Corpus:
{sample}
""",

        "Narrative Story": f"""
You are an ecolinguist applying Arran Stibbe's Stories We Live By 
framework from his 2015 book Ecolinguistics: Language, Ecology and 
the Stories We Live By.

Analyse the corpus below and write a short story of 3 to 5 sentences 
that reveals the NARRATIVE STORY — the underlying cause and effect 
chains in the text, who is constructed as active and who as passive, 
who has agency, and what story arc the text follows overall.

Ground every sentence in specific evidence from the corpus.
Write in plain English. Do not use bullet points.

Corpus:
{sample}
""",
    }

    stories = {}

    for story_type, prompt in stibbe_prompts.items():

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            stories[story_type] = response.content[0].text.strip()

        except Exception as e:
            stories[story_type] = f"Error generating this story: {str(e)}"

    return stories


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌍 Advanced Eco-CDA Analyzer")

st.subheader(
    "Computational Critical Discourse Analysis Tool based on Arran Stibbe's Stories We Live By"
)

st.markdown(
    """
### Features

- Corpus Analysis
- Frequency Analysis
- Concordance / KWIC Analysis
- Stibbe's Classification
- Sentiment Analysis
- Named Entity Recognition
- N-gram Analysis
- Word Clouds
- CDA Interpretation
- Stories We Live By
- Downloadable Results
"""
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📂 Input Options")

input_method = st.sidebar.radio(
    "Choose Input Method",
    [
        "Upload CSV File",
        "Paste Text Directly"
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# Data Input
# ─────────────────────────────────────────────────────────────────────────────
df = None
all_text = ""

if input_method == "Upload CSV File":

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info("Please upload a CSV dataset to begin.")

        st.write("### Required CSV Format")

        st.dataframe(
            pd.DataFrame(
                {
                    "date": ["2024-06-01"],
                    "headline": ["Floods devastate Sindh"],
                    "article_text": [
                        "Pakistan is battling catastrophic floods causing devastation."
                    ],
                }
            )
        )

        st.stop()

    df = pd.read_csv(uploaded_file)

    if "article_text" not in df.columns:

        st.error("Dataset must contain an article_text column.")
        st.stop()

    df["article_text"] = df["article_text"].fillna("").astype(str)

    all_text = " ".join(df["article_text"])

    st.success("CSV uploaded successfully!")

    st.dataframe(df.head())

else:

    pasted_text = st.text_area(
        "Paste your text here (Maximum 100,000 words)",
        height=300,
    )

    if not pasted_text.strip():
        st.info("Please paste text to begin analysis.")
        st.stop()

    wc = count_words(pasted_text)

    if wc > 100000:

        st.error(
            f"Word limit exceeded. Current count: {wc}"
        )

        st.stop()

    st.success(f"Text pasted successfully! ({wc} words)")

    df = pd.DataFrame(
        {
            "article_text": [pasted_text]
        }
    )

    all_text = pasted_text

# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Processing corpus..."):

    tokens = preprocess(all_text)

if not tokens:

    st.error("No valid text extracted.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────
rows = []

with st.spinner("Running discourse classification..."):

    for article in df["article_text"]:

        doc = nlp(article)

        for sent in doc.sents:

            sentence = sent.text.strip()

            cats = classify(sentence)

            sentiment = get_sentiment(sentence)

            if cats:

                rows.append(
                    {
                        "Sentence": sentence,
                        "Category": ", ".join(cats),
                        "Sentiment": sentiment,
                    }
                )

class_df = pd.DataFrame(rows)

cat_counts = Counter()

if not class_df.empty:

    for c in class_df["Category"]:
        cat_counts.update(c.split(", "))

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "📊 Corpus Overview",
        "📈 Frequency",
        "🔍 KWIC",
        "🧠 Stibbe's Classification",   # ← renamed from Classification
        "😊 Sentiment",
        "🏷️ Named Entities",
        "🌍 CDA Insights",
        "📖 Stories We Live By",         # ← new tab
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    st.header("Corpus Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Documents", len(df))
    c2.metric("Processed Words", len(tokens))
    c3.metric("Unique Words", len(set(tokens)))
    c4.metric("Sentences", len(list(nlp(all_text).sents)))

    st.write("### Text Preview")

    st.text_area(
        "Preview",
        all_text[:3000],
        height=250,
        disabled=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    st.header("Frequency Analysis")

    freq = word_freq(tokens)

    freq_df = pd.DataFrame(
        freq.most_common(20),
        columns=["Word", "Frequency"]
    )

    st.write("### Top 20 Words")

    st.dataframe(freq_df)

    fig = px.bar(
        freq_df,
        x="Word",
        y="Frequency",
        title="Top Frequent Words"
    )

    st.plotly_chart(fig, use_container_width=True)

    # WORD CLOUD
    st.write("### Word Cloud")

    wc_obj = WordCloud(
        width=1200,
        height=500,
        background_color="white"
    ).generate(" ".join(tokens))

    fig_wc, ax = plt.subplots(figsize=(14, 6))

    ax.imshow(wc_obj, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig_wc)

    # BIGRAMS
    st.write("### Top Bigrams")

    bigram_df = generate_ngrams(tokens, n=2)

    st.dataframe(bigram_df)

    # TRIGRAMS
    st.write("### Top Trigrams")

    trigram_df = generate_ngrams(tokens, n=3)

    st.dataframe(trigram_df)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3
# ─────────────────────────────────────────────────────────────────────────────
with tab3:

    st.header("KWIC Concordance Analysis")

    kw = st.text_input("Enter keyword")

    if kw:

        hits = concordance(all_text, kw)

        if hits:

            kwic_df = pd.DataFrame(
                hits,
                columns=[
                    "Left Context",
                    "Keyword",
                    "Right Context"
                ]
            )

            st.dataframe(
                kwic_df,
                use_container_width=True
            )

        else:
            st.warning("No occurrences found.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  —  Stibbe's Classification (renamed)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:

    st.header("Stibbe's Classification")

    if not class_df.empty:

        st.dataframe(
            class_df,
            use_container_width=True
        )

        chart_df = pd.DataFrame(
            {
                "Category": list(cat_counts.keys()),
                "Count": list(cat_counts.values()),
            }
        )

        fig_pie = px.pie(
            chart_df,
            names="Category",
            values="Count",
            title="Distribution of Stibbe Categories"
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        st.download_button(
            "📥 Download Classification Results",
            data=class_df.to_csv(index=False),
            file_name="classification_results.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5
# ─────────────────────────────────────────────────────────────────────────────
with tab5:

    st.header("Sentiment Analysis")

    if not class_df.empty:

        sentiment_counts = (
            class_df["Sentiment"]
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = [
            "Sentiment",
            "Count"
        ]

        fig_sent = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            title="Sentiment Distribution"
        )

        st.plotly_chart(
            fig_sent,
            use_container_width=True
        )

        st.dataframe(sentiment_counts)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6
# ─────────────────────────────────────────────────────────────────────────────
with tab6:

    st.header("Named Entity Recognition")

    entity_df = extract_entities(all_text)

    if not entity_df.empty:

        st.dataframe(
            entity_df,
            use_container_width=True
        )

        top_entities = entity_df.head(15)

        fig_ent = px.bar(
            top_entities,
            x="Entity",
            y="Frequency",
            color="Type",
            title="Top Named Entities"
        )

        st.plotly_chart(
            fig_ent,
            use_container_width=True
        )

    else:

        st.warning("No named entities detected.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7
# ─────────────────────────────────────────────────────────────────────────────
with tab7:

    st.header("CDA Insights")

    if cat_counts:

        INSIGHTS = {

            "Identity": (
                "Identity discourse dominates the corpus, suggesting strong focus on social actors, institutions, governments, and affected communities."
            ),

            "Ideology": (
                "Ideological framing reflects governance, development, sustainability, and climate responsibility narratives."
            ),

            "Evaluation": (
                "Evaluative language indicates emotional and ideological positioning within environmental reporting."
            ),

            "Metaphor": (
                "Metaphorical framing constructs environmental crises through conflict or disaster-related imagery."
            ),

            "Erasure": (
                "Erasure patterns may obscure responsibility through passive constructions."
            ),
        }

        for cat, msg in INSIGHTS.items():

            if cat in cat_counts:
                st.info(f"**{cat}:** {msg}")

    else:

        st.warning("No discourse patterns detected.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8  —  Stories We Live By (new)
# ─────────────────────────────────────────────────────────────────────────────
with tab8:

    st.header("📖 Stories We Live By")

    st.markdown(
        """
        This feature generates short stories directly from your corpus,
        each grounded in one of Arran Stibbe's analytical categories
        from *Ecolinguistics: Language, Ecology and the Stories We Live By* (2015).

        Every story is corpus-driven — the language, patterns and absences
        in your text determine what each story reveals.
        """
    )

    st.info(
        "**Theoretical basis:** Arran Stibbe (2015) — "
        "*Ecolinguistics: Language, Ecology and the Stories We Live By*"
    )

    # Story type descriptions shown before generation
    STORY_DESCRIPTIONS = {
        "Framing Story":   "How the topic is conceptually bounded and what assumptions are built in.",
        "Salience Story":  "What is foregrounded and made visible versus pushed to the background.",
        "Metaphor Story":  "The conceptual metaphors structuring perception and what they reveal or conceal.",
        "Identity Story":  "How groups and individuals are constructed — who speaks, who is spoken about.",
        "Erasure Story":   "What voices, perspectives and realities are systematically absent.",
        "Narrative Story": "The cause-effect chains, who has agency, and the overall story arc.",
    }

    # Colour palette — one per story type
    STORY_COLORS = {
        "Framing Story":   "#e8f4f8",
        "Salience Story":  "#f0f8e8",
        "Metaphor Story":  "#fef9e7",
        "Identity Story":  "#fce8f0",
        "Erasure Story":   "#f0e8f8",
        "Narrative Story": "#e8f8f0",
    }

    STORY_ICONS = {
        "Framing Story":   "🖼️",
        "Salience Story":  "🔦",
        "Metaphor Story":  "🌊",
        "Identity Story":  "👥",
        "Erasure Story":   "🔇",
        "Narrative Story": "📜",
    }

    # Show story type descriptions before the button is clicked
    st.write("### The Six Story Types")

    cols = st.columns(2)

    for i, (stype, desc) in enumerate(STORY_DESCRIPTIONS.items()):

        with cols[i % 2]:

            icon = STORY_ICONS[stype]
            color = STORY_COLORS[stype]

            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    padding: 12px 16px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    border-left: 4px solid #888;
                ">
                    <strong>{icon} {stype}</strong><br>
                    <small>{desc}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("---")

    # Generate button
    if st.button(
        "✨ Generate Stories from Corpus",
        type="primary",
        use_container_width=True,
    ):
        # Check API key exists
        if "ANTHROPIC_API_KEY" not in st.secrets:
            st.error(
                "Anthropic API key not found. "
                "Please add ANTHROPIC_API_KEY to your .streamlit/secrets.toml file."
            )

        else:
            with st.spinner(
                "Reading your corpus through Stibbe's framework... "
                "Generating six stories, please wait."
            ):
                stories = generate_stibbe_stories(all_text)

            st.success("Six stories extracted from your corpus.")

            st.write("### Stories from the Corpus")

            for story_type, story_text in stories.items():

                icon  = STORY_ICONS[story_type]
                color = STORY_COLORS[story_type]
                desc  = STORY_DESCRIPTIONS[story_type]

                st.markdown(
                    f"""
                    <div style="
                        background-color: {color};
                        padding: 20px 24px;
                        border-radius: 10px;
                        margin: 12px 0;
                        border-left: 5px solid #555;
                    ">
                        <h4 style="margin: 0 0 4px 0;">
                            {icon} {story_type}
                        </h4>
                        <p style="
                            margin: 0 0 10px 0;
                            font-size: 0.82em;
                            color: #666;
                        ">
                            {desc}
                        </p>
                        <p style="
                            margin: 0;
                            font-style: italic;
                            line-height: 1.7;
                        ">
                            {story_text}
                        </p>
                        <p style="
                            margin: 10px 0 0 0;
                            font-size: 0.75em;
                            color: #999;
                        ">
                            📚 Stibbe, A. (2015).
                            <em>Ecolinguistics: Language, Ecology and
                            the Stories We Live By.</em> Routledge.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Download all stories as a text file
            stories_text = ""

            for stype, stext in stories.items():
                stories_text += f"{stype}\n{'─' * 40}\n{stext}\n\n"

            stories_text += (
                "─" * 40
                + "\nTheoretical basis: Stibbe, A. (2015). "
                "Ecolinguistics: Language, Ecology and the "
                "Stories We Live By. Routledge.\n"
            )

            st.download_button(
                label="📥 Download All Stories",
                data=stories_text,
                file_name="stories_we_live_by.txt",
                mime="text/plain",
                use_container_width=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")

st.caption(
    "Advanced Eco-CDA Analyzer • Streamlit + NLP + Ecolinguistics • "
    "Based on Arran Stibbe's Stories We Live By (2015)"
)
