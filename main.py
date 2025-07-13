import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
import asyncio

# Load .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in your .env file.")
    st.stop()

# Set up Gemini-compatible client
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define the Agent
agent = Agent(
    name="TranslatorAgent",
    instructions="""
    You are a smart multilingual translator.
    Your task:
    1. Detect the input language
    2. Translate the sentence into the user-specified target language
    3. Return only the translated sentence (no extra explanation)

    If the input is in Roman Urdu, treat it as Urdu and translate properly.
    """
)

# --- Streamlit UI Starts Here ---

st.set_page_config(page_title="🌍 AI Translator", layout="centered")
st.title("🌐 Multilingual Translator Agent")
st.caption("Translate anything to your desired language using AI")

st.markdown("#### ✏️ Enter your sentence below:")

input_text = st.text_area("Input Sentence", placeholder="Enter a sentence you want to translate", height=100)

target_lang = st.selectbox("🌐 Translate to:", [
    "English", "Urdu", "Arabic", "French", "Hindi", "Chinese", "German"
])

if st.button("🚀 Translate Now"):
    if not input_text.strip():
        st.warning("Please enter a sentence to translate.")
    else:
        with st.spinner("Translating..."):
            final_prompt = f"Sentence: {input_text}\nTranslate to: {target_lang}"
            try:
                response = asyncio.run(Runner.run(agent, input=final_prompt, run_config=config))
                st.success(f"✅ Translated to {target_lang}:")
                st.markdown(f"**{response}**")
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ by Samra using Gemini + Streamlit")
