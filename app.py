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
    "battle", "fight", "combat", "enemy", "tsunami",
]

EVALUATION_WORDS = [
    "devastating", "catastrophic", "severe", "tragic",
    "dangerous", "massive", "horrific", "destructive",
]

IDENTITY_WORDS = [
    "government", "victims", "families", "citizens", "farmers",
    "authorities", "communities", "people", "minister", "official",
    "community", "citizen",
]

IDEOLOGY_WORDS = [
    "development", "progress", "responsibility", "policy",
    "national", "economic", "climate", "sustainability",
]

ERASURE_PATTERNS = [
    "were displaced", "was destroyed", "were affected",
    "was damaged", "lost their homes",
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
    words = [token.lemma_.lower() for token in doc if token.is_alpha]
    s = " ".join(words)
    cats = []

    metaphor_patterns = ["battle against", "fight against", "war on", "wave of"]
    if any(p in s for p in metaphor_patterns):
        cats.append("Metaphor")
    elif any(w in words for w in METAPHOR_WORDS):
        climate_context = ["climate", "flood", "disaster", "environment", "crisis"]
        if any(c in words for c in climate_context):
            cats.append("Metaphor")

    if any(w in words for w in EVALUATION_WORDS):
        cats.append("Evaluation")
    if any(w in words for w in IDENTITY_WORDS):
        cats.append("Identity")
    if any(w in words for w in IDEOLOGY_WORDS):
        cats.append("Ideology")

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
        left  = text[start:match.start()]
        kw    = text[match.start():match.end()]
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
    grams     = ngrams(tokens, n)
    gram_freq = Counter(grams)
    return pd.DataFrame(
        [(" ".join(k), v) for k, v in gram_freq.most_common(top_k)],
        columns=["Phrase", "Frequency"]
    )


def extract_entities(text):
    doc = nlp(text)
    entities  = [(ent.text, ent.label_) for ent in doc.ents]
    entity_df = pd.DataFrame(entities, columns=["Entity", "Type"])
    if entity_df.empty:
        return entity_df
    return entity_df.value_counts().reset_index(name="Frequency")


# ─────────────────────────────────────────────────────────────────────────────
# Stories We Live By — Shared Constants
# ─────────────────────────────────────────────────────────────────────────────

STORY_DESCRIPTIONS = {
    "Framing Story":   "How the topic is conceptually bounded and what assumptions are built in.",
    "Salience Story":  "What is foregrounded and made visible versus pushed to the background.",
    "Metaphor Story":  "The conceptual metaphors structuring perception and what they reveal or conceal.",
    "Identity Story":  "How groups and individuals are constructed — who speaks, who is spoken about.",
    "Erasure Story":   "What voices, perspectives and realities are systematically absent.",
    "Narrative Story": "The cause-effect chains, who has agency, and the overall story arc.",
}

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


# ─────────────────────────────────────────────────────────────────────────────
# Option 1 — Rule-Based Story Generation (no AI)
# ─────────────────────────────────────────────────────────────────────────────

def generate_stibbe_stories_no_ai(corpus_text, class_df, tokens):
    """
    Generates stories purely from corpus evidence.
    Uses classification results already computed by the tool.
    No external API. Fully reproducible every run.
    """

    top_words     = Counter(tokens).most_common(10)
    top_word_list = ", ".join([w for w, _ in top_words[:5]])

    def get_sentences(category):
        if class_df.empty:
            return []
        mask = class_df["Category"].str.contains(category, na=False)
        return class_df[mask]["Sentence"].head(2).tolist()

    metaphor_sents = get_sentences("Metaphor")
    erasure_sents  = get_sentences("Erasure")
    identity_sents = get_sentences("Identity")
    eval_sents     = get_sentences("Evaluation")

    total_sents  = len(class_df)
    word_count   = len(tokens)
    unique_words = len(set(tokens))

    w0 = top_words[0][0] if len(top_words) > 0 else "the subject"
    w1 = top_words[1][0] if len(top_words) > 1 else "related themes"
    w2 = top_words[2][0] if len(top_words) > 2 else "key concepts"
    f0 = top_words[0][1] if len(top_words) > 0 else 0

    stories = {}

    # FRAMING STORY
    stories["Framing Story"] = (
        f"The corpus constructs its subject through a dominant vocabulary "
        f"centred on '{w0}', '{w1}' and '{w2}', which together appear "
        f"across {word_count} processed words. "
        f"These recurring terms reveal a conceptual frame in which "
        f"'{w0}' functions as the primary lens through which events "
        f"and actors are understood. "
        f"The text takes for granted a world organised around "
        f"{top_word_list} — the assumptions embedded in this framing "
        f"shape what questions are asked and which remain invisible."
    )

    # SALIENCE STORY
    stories["Salience Story"] = (
        f"The word '{w0}' alone appears {f0} times in this corpus, "
        f"making it the single most salient term in the text. "
        f"The top five content words — {top_word_list} — collectively "
        f"dominate the textual landscape, drawing the reader's attention "
        f"repeatedly toward these themes. "
        f"Across {unique_words} unique words, the concentration of "
        f"meaning around this narrow cluster suggests that much of the "
        f"topic's complexity is pushed to the periphery or left unspoken."
    )

    # METAPHOR STORY
    if metaphor_sents:
        metaphor_evidence = (
            f'The corpus contains metaphorical constructions such as: '
            f'"{metaphor_sents[0][:120]}". '
        )
        if len(metaphor_sents) > 1:
            metaphor_evidence += f'A further example is: "{metaphor_sents[1][:120]}". '
    else:
        metaphor_evidence = (
            f"No strong explicit metaphor patterns were flagged, "
            f"yet the dominance of '{w0}' and '{w1}' suggests "
            f"an implicit conceptual metaphor structuring the discourse. "
        )

    stories["Metaphor Story"] = (
        metaphor_evidence
        + "These metaphorical choices are not neutral — they invite "
        "the reader to perceive the subject through a particular "
        "emotional and ideological lens, normalising some responses "
        "while foreclosing others."
    )

    # IDENTITY STORY
    if identity_sents:
        identity_evidence = (
            f'Social actors are constructed in sentences such as: '
            f'"{identity_sents[0][:120]}". '
        )
    else:
        identity_evidence = (
            f"Identity construction in this corpus operates through "
            f"the repeated use of terms like '{w0}' and '{w1}'. "
        )

    stories["Identity Story"] = (
        f"Across {total_sents} classified sentences, the corpus "
        f"consistently positions certain groups as active agents "
        f"and others as passive recipients of action. "
        + identity_evidence
        + "Those with institutional power are named and given voice, "
        "while affected communities are referred to in collective "
        "terms — present in the text but rarely speaking within it."
    )

    # ERASURE STORY
    if erasure_sents:
        erasure_evidence = (
            f'Passive constructions such as "{erasure_sents[0][:120]}" '
            f"remove human agency from the account. "
        )
    else:
        erasure_evidence = (
            f"Erasure in this corpus operates subtly — rather than "
            f"explicit passive constructions, it works through "
            f"the sheer absence of certain voices and perspectives. "
        )

    stories["Erasure Story"] = (
        erasure_evidence
        + "What is systematically absent is as significant as "
        "what is present: the lived experience of those most "
        "affected, their own words, their own framing of events — "
        "none of these appear in the top vocabulary of this corpus. "
        "Responsibility is distributed unevenly, and in places, "
        "dissolved entirely into natural or abstract forces."
    )

    # NARRATIVE STORY
    if eval_sents:
        narrative_evidence = (
            f'The evaluative framing in sentences such as '
            f'"{eval_sents[0][:120]}" '
            f"signals the emotional direction of the narrative arc. "
        )
    else:
        narrative_evidence = (
            f"The narrative arc is carried primarily by the "
            f"high-frequency vocabulary around '{w0}' and '{w1}'. "
        )

    stories["Narrative Story"] = (
        f"The corpus constructs a narrative in which '{w0}' and "
        f"'{w1}' are the primary drivers of events. "
        + narrative_evidence
        + f"Across {total_sents} discourse-classified sentences, "
        "agency is distributed unevenly: some actors cause, "
        "others suffer. The story arc moves from crisis to "
        "institutional response, with structural causes "
        "acknowledged briefly before receding into the background."
    )

    return stories


# ─────────────────────────────────────────────────────────────────────────────
# Option 2 — AI-Enhanced Story Generation (Claude API)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_corpus_sample(text, max_chars=4000):
    """Balanced sample from beginning, middle and end of corpus."""
    if len(text) <= max_chars:
        return text
    third = max_chars // 3
    return (
        text[:third]
        + "\n...\n"
        + text[len(text) // 2: len(text) // 2 + third]
        + "\n...\n"
        + text[-third:]
    )


def generate_stibbe_stories_ai(corpus_text: str):
    """
    Calls Claude once per Stibbe story type.
    Returns a dictionary of {story_type: story_text}.
    """

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    sample = prepare_corpus_sample(corpus_text)

    base = (
        "You are an ecolinguist applying Arran Stibbe's Stories We Live By "
        "framework from Ecolinguistics: Language, Ecology and the Stories We "
        "Live By (2015).\n\n"
        "Write a short story of 3 to 5 sentences.\n"
        "Ground every sentence in specific evidence from the corpus.\n"
        "Write in plain English. Do not use bullet points.\n\n"
        f"Corpus:\n{sample}"
    )

    story_instructions = {
        "Framing Story": (
            "Reveal the FRAMING STORY — how the topic is conceptually bounded, "
            "what assumptions are built into the presentation, and what kind of "
            "world the text takes for granted."
        ),
        "Salience Story": (
            "Reveal the SALIENCE STORY — what is foregrounded and made highly "
            "visible, and what is pushed into the background or mentioned only briefly."
        ),
        "Metaphor Story": (
            "Reveal the METAPHOR STORY — the conceptual metaphors structuring how "
            "the topic is perceived, what they reveal about underlying attitudes, "
            "and what they conceal."
        ),
        "Identity Story": (
            "Reveal the IDENTITY STORY — how groups, communities, institutions and "
            "individuals are constructed and positioned, who is given a voice and "
            "who is spoken about but never heard."
        ),
        "Erasure Story": (
            "Reveal the ERASURE STORY — what voices, perspectives, living beings or "
            "realities are systematically absent or written out entirely, and what "
            "effect this absence creates."
        ),
        "Narrative Story": (
            "Reveal the NARRATIVE STORY — the underlying cause and effect chains, "
            "who is constructed as active and who as passive, who has agency, and "
            "what story arc the text follows overall."
        ),
    }

    stories = {}

    for story_type, instruction in story_instructions.items():
        prompt = base.replace(
            "Write a short story of 3 to 5 sentences.",
            f"Write a short story of 3 to 5 sentences that {instruction}"
        )
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            stories[story_type] = response.content[0].text.strip()
        except Exception as e:
            stories[story_type] = f"Error generating this story: {str(e)}"

    return stories


# ─────────────────────────────────────────────────────────────────────────────
# Shared Story Display Function
# ─────────────────────────────────────────────────────────────────────────────

def display_stories(stories, mode_label):
    """Renders all six story cards and a download button."""

    st.success(f"Six stories generated — {mode_label}")
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
                <h4 style="margin: 0 0 4px 0;">{icon} {story_type}</h4>
                <p style="margin: 0 0 10px 0; font-size: 0.82em; color: #666;">
                    {desc}
                </p>
                <p style="margin: 0; font-style: italic; line-height: 1.7;">
                    {story_text}
                </p>
                <p style="margin: 10px 0 0 0; font-size: 0.75em; color: #999;">
                    📚 Stibbe, A. (2015).
                    <em>Ecolinguistics: Language, Ecology and
                    the Stories We Live By.</em> Routledge.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Build download text
    stories_text = ""
    for stype, stext in stories.items():
        stories_text += f"{stype}\n{'─' * 40}\n{stext}\n\n"
    stories_text += (
        "─" * 40
        + f"\nGeneration mode: {mode_label}"
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
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌍 Advanced Eco-CDA Analyzer")

st.subheader(
    "Computational Critical Discourse Analysis Tool based on "
    "Arran Stibbe's Stories We Live By"
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
    ["Upload CSV File", "Paste Text Directly"]
)

# ─────────────────────────────────────────────────────────────────────────────
# Data Input
# ─────────────────────────────────────────────────────────────────────────────
df       = None
all_text = ""

if input_method == "Upload CSV File":

    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV dataset to begin.")
        st.write("### Required CSV Format")
        st.dataframe(
            pd.DataFrame({
                "date":         ["2024-06-01"],
                "headline":     ["Floods devastate Sindh"],
                "article_text": ["Pakistan is battling catastrophic floods causing devastation."],
            })
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
        st.error(f"Word limit exceeded. Current count: {wc}")
        st.stop()

    st.success(f"Text pasted successfully! ({wc} words)")
    df       = pd.DataFrame({"article_text": [pasted_text]})
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
            sentence  = sent.text.strip()
            cats      = classify(sentence)
            sentiment = get_sentiment(sentence)
            if cats:
                rows.append({
                    "Sentence":  sentence,
                    "Category":  ", ".join(cats),
                    "Sentiment": sentiment,
                })

class_df   = pd.DataFrame(rows)
cat_counts = Counter()

if not class_df.empty:
    for c in class_df["Category"]:
        cat_counts.update(c.split(", "))

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Corpus Overview",
    "📈 Frequency",
    "🔍 KWIC",
    "🧠 Stibbe's Classification",
    "😊 Sentiment",
    "🏷️ Named Entities",
    "🌍 CDA Insights",
    "📖 Stories We Live By",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Corpus Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents",       len(df))
    c2.metric("Processed Words", len(tokens))
    c3.metric("Unique Words",    len(set(tokens)))
    c4.metric("Sentences",       len(list(nlp(all_text).sents)))
    st.write("### Text Preview")
    st.text_area("Preview", all_text[:3000], height=250, disabled=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Frequency Analysis")
    freq    = word_freq(tokens)
    freq_df = pd.DataFrame(freq.most_common(20), columns=["Word", "Frequency"])

    st.write("### Top 20 Words")
    st.dataframe(freq_df)

    fig = px.bar(freq_df, x="Word", y="Frequency", title="Top Frequent Words")
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Word Cloud")
    wc_obj = WordCloud(width=1200, height=500, background_color="white").generate(
        " ".join(tokens)
    )
    fig_wc, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc_obj, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)

    st.write("### Top Bigrams")
    st.dataframe(generate_ngrams(tokens, n=2))

    st.write("### Top Trigrams")
    st.dataframe(generate_ngrams(tokens, n=3))

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
                hits, columns=["Left Context", "Keyword", "Right Context"]
            )
            st.dataframe(kwic_df, use_container_width=True)
        else:
            st.warning("No occurrences found.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  —  Stibbe's Classification
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Stibbe's Classification")
    if not class_df.empty:
        st.dataframe(class_df, use_container_width=True)
        chart_df = pd.DataFrame({
            "Category": list(cat_counts.keys()),
            "Count":    list(cat_counts.values()),
        })
        fig_pie = px.pie(
            chart_df, names="Category", values="Count",
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
            class_df["Sentiment"].value_counts().reset_index()
        )
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig_sent = px.pie(
            sentiment_counts, names="Sentiment", values="Count",
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig_sent, use_container_width=True)
        st.dataframe(sentiment_counts)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.header("Named Entity Recognition")
    entity_df = extract_entities(all_text)
    if not entity_df.empty:
        st.dataframe(entity_df, use_container_width=True)
        fig_ent = px.bar(
            entity_df.head(15), x="Entity", y="Frequency", color="Type",
            title="Top Named Entities"
        )
        st.plotly_chart(fig_ent, use_container_width=True)
    else:
        st.warning("No named entities detected.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.header("CDA Insights")
    if cat_counts:
        INSIGHTS = {
            "Identity":   "Identity discourse dominates the corpus, suggesting strong focus on social actors, institutions, governments, and affected communities.",
            "Ideology":   "Ideological framing reflects governance, development, sustainability, and climate responsibility narratives.",
            "Evaluation": "Evaluative language indicates emotional and ideological positioning within environmental reporting.",
            "Metaphor":   "Metaphorical framing constructs environmental crises through conflict or disaster-related imagery.",
            "Erasure":    "Erasure patterns may obscure responsibility through passive constructions.",
        }
        for cat, msg in INSIGHTS.items():
            if cat in cat_counts:
                st.info(f"**{cat}:** {msg}")
    else:
        st.warning("No discourse patterns detected.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8  —  Stories We Live By
# ─────────────────────────────────────────────────────────────────────────────
with tab8:

    st.header("📖 Stories We Live By")

    st.markdown(
        """
        This feature generates six short stories directly from your corpus,
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

    # ── Story type preview cards ──────────────────────────────────────────
    st.write("### The Six Story Types")

    cols = st.columns(2)
    for i, (stype, desc) in enumerate(STORY_DESCRIPTIONS.items()):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div style="
                    background-color: {STORY_COLORS[stype]};
                    padding: 12px 16px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    border-left: 4px solid #888;
                ">
                    <strong>{STORY_ICONS[stype]} {stype}</strong><br>
                    <small>{desc}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("---")

    # ── Mode toggle ───────────────────────────────────────────────────────
    st.write("### Generation Mode")

    use_ai = st.toggle(
        "✨ Enhance stories with AI interpretation",
        value=False,
        help=(
            "OFF — Stories are generated entirely from your corpus using "
            "rule-based NLP. Fully reproducible. No internet required. "
            "Recommended for academic use.\n\n"
            "ON — Stories are interpreted by Claude AI for richer, more "
            "nuanced output. Requires an Anthropic API key."
        )
    )

    if use_ai:
        st.markdown(
            """
            <div style="
                background-color: #fff8e1;
                padding: 10px 16px;
                border-radius: 8px;
                border-left: 4px solid #f9a825;
                margin-bottom: 10px;
            ">
                <strong>✨ AI Mode active</strong> — Claude will read a sample
                of your corpus and write interpretive stories. Results may vary
                slightly between runs. Requires
                <code>ANTHROPIC_API_KEY</code> in your secrets.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                background-color: #e8f5e9;
                padding: 10px 16px;
                border-radius: 8px;
                border-left: 4px solid #388e3c;
                margin-bottom: 10px;
            ">
                <strong>🔬 Rule-Based Mode active</strong> — Stories are built
                directly from your corpus evidence using NLP classification
                results. Fully reproducible. No API needed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("---")

    # ── Generate button ───────────────────────────────────────────────────
    if st.button(
        "✨ Generate Stories from Corpus",
        type="primary",
        use_container_width=True,
    ):
        if use_ai:
            if "ANTHROPIC_API_KEY" not in st.secrets:
                st.error(
                    "Anthropic API key not found. "
                    "Please add ANTHROPIC_API_KEY to your "
                    ".streamlit/secrets.toml file, or switch to "
                    "Rule-Based Mode."
                )
            else:
                with st.spinner(
                    "Reading your corpus through Stibbe's framework "
                    "with AI — generating six stories, please wait..."
                ):
                    stories = generate_stibbe_stories_ai(all_text)

                display_stories(
                    stories,
                    mode_label="AI-enhanced interpretation (Claude)"
                )
        else:
            with st.spinner("Extracting stories from corpus evidence..."):
                stories = generate_stibbe_stories_no_ai(
                    all_text, class_df, tokens
                )
            display_stories(
                stories,
                mode_label="Rule-based corpus analysis (no AI)"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")

st.caption(
    "Advanced Eco-CDA Analyzer • Streamlit + NLP + Ecolinguistics • "
    "Based on Arran Stibbe's Stories We Live By (2015)"
)
