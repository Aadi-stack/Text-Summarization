import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from gtts import gTTS
import os

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and URL (YT or Website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    language = st.selectbox("Select Language", ["en", "es", "fr", "de", "hi"])

generic_url = st.text_input("URL", label_visibility="collapsed")

## Updated Model (Fixed Deprecated Issue)
llm = ChatGroq(model="mixtral-8x7b", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def get_youtube_transcript(video_url):
    try:
        video_id = video_url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        return None

if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
    else:
        try:
            with st.spinner("Fetching and summarizing..."):
                if "youtube.com" in generic_url:
                    docs_text = get_youtube_transcript(generic_url)
                    if not docs_text:
                        st.error("Failed to retrieve YouTube transcript.")
                        st.stop()
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                  headers={"User-Agent": "Mozilla/5.0"})
                    docs = loader.load()
                    docs_text = docs[0].page_content if docs else ""
                
                if not docs_text:
                    st.error("Could not retrieve content from the provided URL.")
                else:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs_text)
                    st.success(output_summary)
                    
                    # Convert summary to speech
                    speech = gTTS(text=output_summary, lang=language)
                    speech.save("summary.mp3")
                    st.audio("summary.mp3", format="audio/mp3")
        except Exception as e:
            st.exception(f"Exception: {e}")
