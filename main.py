from langchain_community.chat_models.openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.multi_prompt import MultiPromptChain
from langchain.chains import LLMRouterChain, LLMChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from audio_recorder_streamlit import audio_recorder
import streamlit as st
from dotenv import load_dotenv
import os, warnings, time, random, requests, asyncio
warnings.filterwarnings("ignore")
from pydub import AudioSegment
from io import BytesIO
from deepgram import Deepgram

load_dotenv()

if 'OPENAI_API_KEY' in st.secrets['secrets']:
    openai_api_key = st.secrets['secrets']['OPENAI_API_KEY']
else:
    openai_api_key = os.getenv('OPENAI_API_KEY')

if 'ANTHROPIC_API_KEY' in st.secrets['secrets']:
    anthropic_api_key = st.secrets['secrets']['ANTHROPIC_API_KEY']
else:
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

if 'GOOGLE_API_KEY' in st.secrets['secrets']:
    gemini_api_key = st.secrets['secrets']['GOOGLE_API_KEY']
else:
    gemini_api_key = os.getenv('GOOGLE_API_KEY')

if 'DEEPGRAM_API_KEY' in st.secrets['secrets']:
    deepgram_api_key = st.secrets['secrets']['DEEPGRAM_API_KEY']
else:
    deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')

openai = ChatOpenAI(model_name='gpt-4o',api_key=openai_api_key,temperature=0.6)
claude = ChatAnthropic(model_name='claude-3-5-sonnet-20240620',api_key=anthropic_api_key)
gemini = ChatGoogleGenerativeAI(api_key=gemini_api_key,model='gemini-1.5-pro')
parser = StrOutputParser()
deepgram = Deepgram(deepgram_api_key)
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"

async def transcribe_audio(audio_file_path: str):
    """Transcribes an audio file using Deepgram API with enhanced accuracy settings."""
    try:
        with open(audio_file_path, "rb") as audio:
            response = await deepgram.transcription.prerecorded(
                {"buffer": audio, "mimetype": "audio/wav"},
                options={
                    "smart_format": True,
                    "punctuate": True,
                    "language": "en",
                    "model": "nova",
                    "diarization": True
                }
            )

            if not response or "results" not in response:
                print("Error: No transcription results received.")
                return None

            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            return transcript
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

# Define Prompt Templates
behavioral_template = (
    "Generate behavioral interview questions based on the following user query:\n{input}\n"
    "These questions should assess a candidate's problem-solving, teamwork, and leadership skills."
)

technical_template = (
    "Generate technical interview questions based on the following user query:\n{input}\n"
    "Ensure the questions evaluate domain-specific knowledge, coding skills, and system design expertise."
)

case_study_template = (
    "Generate case study interview questions based on the following user query:\n{input}\n"
    "Focus on real-world business challenges that test analytical thinking, decision-making, and strategy formulation."
)

situational_template = (
    "Generate situational interview questions based on the following user query:\n{input}\n"
    "The questions should focus on hypothetical workplace scenarios, unexpected challenges, and ethical dilemmas."
)

leadership_template = (
    "Generate leadership and management interview questions based on the following user query:\n{input}\n"
    "The questions should focus on decision-making, team management, conflict resolution, and strategic thinking."
)

problem_solving_template = (
    "Generate problem-solving interview questions based on the following user query:\n{input}\n"
    "The questions should focus on logical thinking, decision-making, and creative problem-solving."
)

ethical_template = (
    "Generate ethical dilemma interview questions based on the following user query:\n{input}\n"
    "The questions should focus on workplace ethics, integrity, and handling sensitive situations."
)

communication_template = (
    "Generate communication and collaboration interview questions based on the following user query:\n{input}\n"
    "The questions should assess teamwork, conflict resolution, and cross-functional collaboration."
)

prompt_info = [
    {
        'name': 'behavioral',
        'description': "Generates behavioral interview questions based on user's query.",
        'template': behavioral_template
    },
    {
        'name': 'technical',
        'description': "Generates technical interview questions based on user's query.",
        'template': technical_template
    },
    {
        'name': 'case study',
        'description': "Generates case study interview questions based on user's query.",
        'template': case_study_template
    },
    {
        'name':'situational',
        'description': "Generates situational interview questions based on user's query.",
        'template': situational_template
    },
    {
        'name': 'leadership',
        'description': "Generates leadership and management interview questions based on user's query.",
        'template': leadership_template
    },
    {
        'name': 'problem-solving',
        'description': "Generates problem-solving interview questions based on user's query.",
        'template': problem_solving_template
    },
    {
        'name': 'ethical',
        'description': "Generates ethical dilemma interview questions based on user's query.",
        'template': ethical_template
    },
    {
        'name': 'communication',
        'description': "Generates communication and collaboration interview questions based on user's query.",
        'template': communication_template
    },
]

destinations_chain = {}

behavioral_prompt = ChatPromptTemplate.from_template(behavioral_template)
technical_prompt = ChatPromptTemplate.from_template(technical_template)
case_study_prompt = ChatPromptTemplate.from_template(case_study_template)
situational_prompt = ChatPromptTemplate.from_template(situational_template)
leadership_prompt = ChatPromptTemplate.from_template(leadership_template)
problem_solving_prompt = ChatPromptTemplate.from_template(problem_solving_template)
ethical_prompt = ChatPromptTemplate.from_template(ethical_template)
communication_prompt = ChatPromptTemplate.from_template(communication_template)

behavioral_chain = LLMChain(llm=openai, prompt=behavioral_prompt)
technical_chain = LLMChain(llm=claude, prompt=technical_prompt)
case_study_chain = LLMChain(llm=gemini, prompt=case_study_prompt)
situational_chain = LLMChain(llm=openai, prompt=situational_prompt)
leadership_chain = LLMChain(llm=claude, prompt=leadership_prompt)
problem_solving_chain = LLMChain(llm=gemini, prompt=problem_solving_prompt)
ethical_chain = LLMChain(llm=openai, prompt=ethical_prompt)
communication_chain = LLMChain(llm=claude, prompt=communication_prompt)

destinations_chain['behavioral'] = behavioral_chain
destinations_chain['technical'] = technical_chain
destinations_chain['case study'] = case_study_chain
destinations_chain['situational'] = situational_chain
destinations_chain['leadership'] = leadership_chain
destinations_chain['problem-solving'] = problem_solving_chain
destinations_chain['ethical'] = ethical_chain
destinations_chain['communication'] = communication_chain

evaluation_prompt = PromptTemplate(
    input_variables=["question", "user_answer"],
    template="Evaluate the following answer: {user_answer} for the question: {question}. Provide a score (1-10) and detailed feedback."
)
evaluation_chain = LLMChain(llm=openai, prompt=evaluation_prompt)

default_template = "{input}"
default_prompt = ChatPromptTemplate.from_template(default_template)
default_chain = LLMChain(llm=openai, prompt=default_prompt)

destinations = "\n".join([f"{p_info['name']}: {p_info['description']}" for p_info in prompt_info])

# Define the router
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations)
router_prompt = PromptTemplate(template=router_template,
                               input_variables=['input'],
                               output_parser=RouterOutputParser())

router_chain = LLMRouterChain.from_llm(llm=claude, prompt=router_prompt)

chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destinations_chain,
                         default_chain=default_chain,
                         verbose=True)

# Initialize memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Streamlit UI
st.set_page_config(page_title='AI-Powered Interview Coach', layout='wide')

st.title("🤖 AI-Powered Interview Preparation Guide")
st.sidebar.header("Prepare for Interview")

# User Inputs
role = st.sidebar.text_input("Enter Job Role", value="Data Scientist")
difficulty = st.sidebar.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])
experience = st.sidebar.selectbox("Select Experience Level", ["Beginner", "Intermediate", "Expert"])
user_query = st.sidebar.text_area("Enter your query here:", value="", placeholder="What are you looking for?")
user_answer = st.sidebar.text_area("Enter your answer here:", value="", placeholder="How would you respond to this question?")

use_speech_input = st.sidebar.checkbox("Use Speech Input for Answer")

if use_speech_input:
    st.subheader("🎤 Speak your answer")
    
    # audio_bytes = st.audio(audio_recorder(), format="audio/wav")
    audio_bytes = audio_recorder()

    # # Convert audio data into a WAV file using pydub
    # if audio_input is not None:
    #     # # Convert raw audio data into a WAV file using pydub
    #     # audio_bytes = BytesIO(audio_input.getvalue())
        
    if audio_bytes:
        if isinstance(audio_bytes, bytes):
            audio_file = BytesIO(audio_bytes)

            # Convert to WAV format with correct sample rate (16kHz recommended)
            audio = AudioSegment.from_file(audio_file, format='wav')
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16-bit PCM

            # # Save the corrected WAV file
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            #     audio.export(temp_audio.name, format="wav")
            #     temp_audio_path = temp_audio.name
            
            # Save the processed WAV file
            file_path = "user_answer.wav"
            audio.export(file_path, format="wav")

            transcription = asyncio.run(transcribe_audio(file_path))

            st.write("**Transcription:**")
            st.write(transcription)
            user_answer = transcription

generate_questions = st.sidebar.button("Generate Questions")
questions = ""

# Generate Questions Button
if generate_questions:
    query = str(user_query) + f"\nThe user is a {role} with {experience} experience and wants {difficulty} level interview questions."
    
    with st.spinner("Generating interview questions..."):
        # Progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.25)  # Simulate work being done
            progress_bar.progress(i + 1)

        result = chain.run(input=query)
        questions = result
    
    st.subheader("📝 Generated Interview Questions")
    st.write(result)

evaluate_answers = st.sidebar.button("Evaluate Answer")

# Evaluate Answer Button
if evaluate_answers:
    if user_answer:
        with st.spinner("Evaluating your answer..."):
            # Progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.05)  # Simulate work being done
                progress_bar.progress(i + 1)
            evaluation_result = evaluation_chain.invoke({"question": questions, "user_answer": user_answer})
        
        st.subheader("📊 AI Evaluation Feedback")
        evaluation_result = parser.parse(str(evaluation_result['text']))
        st.write(evaluation_result)
    else:
        st.warning("Please enter your answer before clicking the Evaluate button.")