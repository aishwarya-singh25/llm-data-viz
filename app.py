import base64
import numpy as np
import streamlit as st
from lida import Manager, TextGenerationConfig, llm

openai_api_key = 'sk-' ## add openAI api key here
st.title("Simple viz")

prompt = st.chat_input(
    "Ask your query for the shopping trends dataset"
)

if not prompt: st.stop()

with st.chat_message("question"):
    st.markdown(prompt, unsafe_allow_html=True)

with st.spinner("Thinking..."):
    library = "seaborn"
    lida = Manager(text_gen = llm("openai", api_key=openai_api_key))
    textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo", use_cache=True)
    summary = lida.summarize("shopping_trends.csv", summary_method="default", textgen_config=textgen_config)
    charts = lida.visualize(summary=summary, goal=prompt, textgen_config=textgen_config, library=library)

    # if generation fails, retry five times
    cnt = 5
    while (len(charts) < 1) and (cnt > 0):
        textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="gpt-3.5-turbo", use_cache=False)
        charts = lida.visualize(summary=summary, goal=prompt, textgen_config=textgen_config, library=library)
        cnt -= 1

    # if still fails
    with st.chat_message("response"):
        st.write("You need more money, buy GPT5")

    with st.chat_message("response"):
        st.image(base64.b64decode(charts[0].raster))
