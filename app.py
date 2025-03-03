import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import time
import pyttsx3
from gtts import gTTS
import os

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text from YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text from YT or Website")
st.subheader('Summarize URL')

# Sidebar for API Key Input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    language = st.selectbox("Select Summary Language", ["English", "Hindi", "French", "Spanish", "German"])
    enable_audio = st.checkbox("Enable Voice Summarization")

# URL Input Field
generic_url = st.text_input("Enter URL", label_visibility="collapsed")

# Update Model to gemma-9b
llm = ChatGroq(model="gemma-9b", groq_api_key=groq_api_key)

# Prompt Template
prompt_template = """
Provide a concise summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def get_youtube_title(url, retries=3, delay=5):
    """Fetches YouTube video title with retries."""
    for attempt in range(retries):
        try:
            yt = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            return yt.load()
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    return None

if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the necessary information to proceed.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or website).")
    else:
        try:
            with st.spinner("Processing..."):
                # Load website or YouTube content
                if "youtube.com" in generic_url:
                    docs = get_youtube_title(generic_url)
                    if not docs:
                        st.error("Failed to load YouTube content after multiple attempts.")
                        st.stop()
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False, headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs = loader.load()
                
                # Chain for Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                
                # Display Summary
                st.success(output_summary)
                
                # Voice Summarization (if enabled)
                if enable_audio:
                    tts = gTTS(output_summary, lang=language[:2].lower())  # Convert first 2 chars to language code
                    tts.save("summary_audio.mp3")
                    st.audio("summary_audio.mp3", format="audio/mp3")
        except Exception as e:
            st.exception(f"Exception: {e}")
