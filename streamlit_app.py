import streamlit as st
import os
import tempfile

GENRES = [
    "ambient",
    "blues",
    "classical",
    "country",
    "electronic",
    "folk",
    "hip hop",
    "jazz",
    "latin",
    "lo-fi",
    "metal",
    "pop",
    "r&b",
    "reggae",
    "rock",
]

@st.cache_resource
def load_models():
    from blend_music import generate_from_blend
    from ML_engineering.infer import predict_genre
    return generate_from_blend, predict_genre

generate_from_blend, predict_genre = load_models()

st.markdown("""
<style>
/* ── Background: lavender, black text ── */
[data-testid="stAppViewContainer"] {
    background-color: #D4CBF2;
    color: #000000;
}
[data-testid="stHeader"] {
    background: transparent;
}

/* ── Default text black (on lavender background) ── */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: #000000;
}

/* ── Title ── */
h1 {
    text-align: center;
    letter-spacing: 0.04em;
}

/* ── Tab bar: blue background, white text ── */
[data-testid="stTabs"] button {
    background-color: #92AAE8;
    color: #ffffff !important;
    font-weight: 600;
    font-size: 1rem;
    border-radius: 6px 6px 0 0;
    padding: 6px 20px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #7a95e0;
    color: #ffffff !important;
    border-bottom: 3px solid #D4CBF2;
}

/* ── Buttons: blue background, white text ── */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
    background-color: #92AAE8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background-color: #7a95e0 !important;
}

/* ── Slider accent ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #92AAE8;
}

/* ── Selectbox: blue background, white text ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #92AAE8;
    border-color: #92AAE8;
    color: #ffffff;
}

/* ── Metric value: blue on lavender ── */
[data-testid="stMetricValue"] {
    color: #92AAE8;
    font-size: 1.8rem;
}

/* ── File uploader: blue dashed border, black text ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #92AAE8;
    border-radius: 8px;
    padding: 8px;
}

/* ── Caption: black on lavender ── */
[data-testid="stCaptionContainer"] {
    color: #000000;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #D4CBF2;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.music-bg {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0; pointer-events: none; overflow: visible;
}
.sym {
    position: absolute;
    color: #000000;
    user-select: none;
    font-family: serif;
    line-height: 1;
}
</style>
<div class="music-bg" aria-hidden="true">

  <!-- ── LEFT SIDE ── -->

  <!-- Staff 1 — top left, treble clef + notes -->
  <svg style="position:absolute;top:7%;left:2%;transform:rotate(-6deg)" width="500" height="200" overflow="visible" xmlns="http://www.w3.org/2000/svg">
    <line x1="0" y1="88"  x2="500" y2="88"  stroke="#000" stroke-width="2"/>
    <line x1="0" y1="100" x2="500" y2="100" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="112" x2="500" y2="112" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="124" x2="500" y2="124" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="136" x2="500" y2="136" stroke="#000" stroke-width="2"/>
    <text x="3"   y="138" font-size="80" font-family="serif" fill="#000">&#x1D11E;</text>
    <text x="80"  y="104" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="118" y="129" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="158" y="116" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="198" y="96"  font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="238" y="133" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="278" y="108" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="318" y="120" font-size="38" font-family="serif" fill="#000">&#9836;</text>
    <text x="358" y="96"  font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="398" y="133" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="455" y="108" font-size="38" font-family="serif" fill="#000">&#9834;</text>
  </svg>

  <!-- Staff 2 — mid left, treble clef + notes -->
  <svg style="position:absolute;top:43%;left:1%;transform:rotate(5deg)" width="500" height="200" overflow="visible" xmlns="http://www.w3.org/2000/svg">
    <line x1="0" y1="88"  x2="500" y2="88"  stroke="#000" stroke-width="2"/>
    <line x1="0" y1="100" x2="500" y2="100" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="112" x2="500" y2="112" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="124" x2="500" y2="124" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="136" x2="500" y2="136" stroke="#000" stroke-width="2"/>
    <text x="3"   y="138" font-size="80" font-family="serif" fill="#000">&#x1D11E;</text>
    <text x="80"  y="123" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="120" y="102" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="160" y="139" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="202" y="114" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="242" y="128" font-size="38" font-family="serif" fill="#000">&#9836;</text>
    <text x="285" y="94"  font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="330" y="118" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="378" y="103" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="428" y="133" font-size="38" font-family="serif" fill="#000">&#9834;</text>
  </svg>

  <!-- Staff 3 — bottom left, bass clef + notes -->
  <svg style="position:absolute;top:76%;left:0.5%;transform:rotate(-4deg)" width="500" height="200" overflow="visible" xmlns="http://www.w3.org/2000/svg">
    <line x1="0" y1="88"  x2="500" y2="88"  stroke="#000" stroke-width="2"/>
    <line x1="0" y1="100" x2="500" y2="100" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="112" x2="500" y2="112" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="124" x2="500" y2="124" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="136" x2="500" y2="136" stroke="#000" stroke-width="2"/>
    <text x="3"   y="134" font-size="80" font-family="serif" fill="#000">&#x1D122;</text>
    <text x="80"  y="110" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="120" y="131" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="162" y="98"  font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="205" y="120" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="248" y="100" font-size="38" font-family="serif" fill="#000">&#9836;</text>
    <text x="295" y="135" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="342" y="112" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="392" y="126" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="445" y="98"  font-size="38" font-family="serif" fill="#000">&#9834;</text>
  </svg>

  <!-- ── RIGHT SIDE ── -->

  <!-- Staff 4 — upper right, treble clef + notes -->
  <svg style="position:absolute;top:17%;right:0.5%;transform:rotate(7deg)" width="500" height="200" overflow="visible" xmlns="http://www.w3.org/2000/svg">
    <line x1="0" y1="88"  x2="500" y2="88"  stroke="#000" stroke-width="2"/>
    <line x1="0" y1="100" x2="500" y2="100" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="112" x2="500" y2="112" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="124" x2="500" y2="124" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="136" x2="500" y2="136" stroke="#000" stroke-width="2"/>
    <text x="3"   y="138" font-size="80" font-family="serif" fill="#000">&#x1D11E;</text>
    <text x="80"  y="116" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="120" y="96"  font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="160" y="133" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="202" y="108" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="244" y="124" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="286" y="100" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="328" y="118" font-size="38" font-family="serif" fill="#000">&#9836;</text>
    <text x="372" y="96"  font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="418" y="132" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="462" y="110" font-size="38" font-family="serif" fill="#000">&#9834;</text>
  </svg>

  <!-- Staff 5 — mid right, ALTO CLEF + notes -->
  <svg style="position:absolute;top:53%;right:0.5%;transform:rotate(-5deg)" width="500" height="200" overflow="visible" xmlns="http://www.w3.org/2000/svg">
    <line x1="0" y1="88"  x2="500" y2="88"  stroke="#000" stroke-width="2"/>
    <line x1="0" y1="100" x2="500" y2="100" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="112" x2="500" y2="112" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="124" x2="500" y2="124" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="136" x2="500" y2="136" stroke="#000" stroke-width="2"/>
    <text x="3"   y="138" font-size="70" font-family="serif" fill="#000">&#x1D121;</text>
    <text x="80"  y="104" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="120" y="127" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="162" y="96"  font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="205" y="116" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="248" y="133" font-size="38" font-family="serif" fill="#000">&#9836;</text>
    <text x="292" y="100" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="338" y="122" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="385" y="98"  font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="435" y="130" font-size="38" font-family="serif" fill="#000">&#9834;</text>
  </svg>

  <!-- Staff 6 — bottom right, bass clef + notes -->
  <svg style="position:absolute;top:83%;right:1%;transform:rotate(4deg)" width="500" height="200" overflow="visible" xmlns="http://www.w3.org/2000/svg">
    <line x1="0" y1="88"  x2="500" y2="88"  stroke="#000" stroke-width="2"/>
    <line x1="0" y1="100" x2="500" y2="100" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="112" x2="500" y2="112" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="124" x2="500" y2="124" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="136" x2="500" y2="136" stroke="#000" stroke-width="2"/>
    <text x="3"   y="134" font-size="80" font-family="serif" fill="#000">&#x1D122;</text>
    <text x="80"  y="129" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="120" y="104" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="162" y="120" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="205" y="98"  font-size="38" font-family="serif" fill="#000">&#9836;</text>
    <text x="248" y="136" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="295" y="112" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="345" y="125" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="395" y="100" font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="448" y="134" font-size="38" font-family="serif" fill="#000">&#9834;</text>
  </svg>

  <!-- Staff 7 — right of Generate button, treble clef + notes -->
  <svg style="position:absolute;top:65%;left:37%;transform:rotate(8deg)" width="500" height="200" overflow="visible" xmlns="http://www.w3.org/2000/svg">
    <line x1="0" y1="88"  x2="500" y2="88"  stroke="#000" stroke-width="2"/>
    <line x1="0" y1="100" x2="500" y2="100" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="112" x2="500" y2="112" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="124" x2="500" y2="124" stroke="#000" stroke-width="2"/>
    <line x1="0" y1="136" x2="500" y2="136" stroke="#000" stroke-width="2"/>
    <text x="3"   y="138" font-size="80" font-family="serif" fill="#000">&#x1D11E;</text>
    <text x="80"  y="116" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="120" y="94"  font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="162" y="130" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="205" y="108" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="248" y="122" font-size="38" font-family="serif" fill="#000">&#9836;</text>
    <text x="292" y="98"  font-size="38" font-family="serif" fill="#000">&#9833;</text>
    <text x="338" y="134" font-size="38" font-family="serif" fill="#000">&#9835;</text>
    <text x="385" y="112" font-size="38" font-family="serif" fill="#000">&#9834;</text>
    <text x="435" y="126" font-size="38" font-family="serif" fill="#000">&#9833;</text>
  </svg>



  <!-- Scattered standalone notes -->
  <span class="sym" style="top:2%;left:18%;font-size:3.8rem;transform:rotate(18deg)">&#9835;</span>
  <span class="sym" style="top:19%;left:9%;font-size:3.2rem;transform:rotate(-22deg)">&#9834;</span>
  <span class="sym" style="top:33%;left:4%;font-size:3.5rem;transform:rotate(14deg)">&#9836;</span>
  <span class="sym" style="top:58%;left:8%;font-size:3rem;transform:rotate(-17deg)">&#9833;</span>
  <span class="sym" style="top:68%;left:19%;font-size:3.4rem;transform:rotate(20deg)">&#9835;</span>
  <span class="sym" style="top:90%;left:10%;font-size:3.8rem;transform:rotate(-13deg)">&#9834;</span>
  <span class="sym" style="top:3%;right:16%;font-size:3.6rem;transform:rotate(-19deg)">&#9834;</span>
  <span class="sym" style="top:12%;right:6%;font-size:3.2rem;transform:rotate(15deg)">&#9836;</span>
  <span class="sym" style="top:36%;right:7%;font-size:3.8rem;transform:rotate(-21deg)">&#9833;</span>
  <span class="sym" style="top:47%;right:17%;font-size:3.3rem;transform:rotate(16deg)">&#9835;</span>
  <span class="sym" style="top:70%;right:8%;font-size:3.5rem;transform:rotate(-12deg)">&#9836;</span>
  <span class="sym" style="top:92%;right:14%;font-size:3.2rem;transform:rotate(22deg)">&#9833;</span>
  <span class="sym" style="top:1%;left:40%;font-size:3.4rem;transform:rotate(-15deg)">&#9836;</span>
  <span class="sym" style="top:1%;right:40%;font-size:3rem;transform:rotate(11deg)">&#9833;</span>
  <span class="sym" style="top:95%;left:30%;font-size:3.5rem;transform:rotate(17deg)">&#9835;</span>
  <span class="sym" style="top:95%;right:30%;font-size:3.2rem;transform:rotate(-14deg)">&#9834;</span>

</div>
""", unsafe_allow_html=True)

st.title("GenreBlender: Generative AI Music Mixer 🎵🤖")

tab1, tab2 = st.tabs(["Genre Blender", "Genre Classifier"])

# ── Tab 1: Genre Blender ──────────────────────────────────────────────────────
with tab1:
    st.write("Generate a blend of two genres using AI.")

    col1, col2 = st.columns(2)
    with col1:
        genreA = st.selectbox("Genre A", GENRES, index=GENRES.index("classical"))
    with col2:
        genreB = st.selectbox("Genre B", GENRES, index=GENRES.index("hip hop"))

    alpha = st.slider("Blend (0 = all Genre B, 1 = all Genre A)", 0.0, 1.0, 0.5, 0.05)
    st.caption(f"{int(alpha*100)}% {genreA}  |  {int((1-alpha)*100)}% {genreB}")

    if st.button("Generate", disabled=not (genreA and genreB)):
        with st.spinner("Generating music..."):
            out_dir = tempfile.mkdtemp()
            mp3_path = generate_from_blend(genreA, genreB, alpha, out_dir)

        st.audio(mp3_path, format="audio/mp3")

# ── Tab 2: Genre Classifier ───────────────────────────────────────────────────
with tab2:
    st.write("Upload an audio file to predict its genre.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "flac"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.rsplit('.', 1)[-1]}")

        if st.button("Classify"):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Classifying..."):
                result = predict_genre(tmp_path)

            os.unlink(tmp_path)

            st.subheader("Genre Prediction")
            st.metric("Predicted Genre", result["genre"].capitalize())

            st.subheader("Probabilities")
            probs = dict(sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True))
            st.bar_chart(probs)