import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 모델 설정
LLM_MODEL = "gpt-4.1-mini"
STT_MODEL = "gpt-4o-mini-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "marin"  # 엄마 느낌으로 먼저 써보기 좋은 목소리

# OpenAI 키 확인
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY를 .env 파일에 넣어주세요.")
    st.stop()

client = OpenAI()


def build_system_instruction(sex, age):
    return f"""
# Rule
- Role: 당신은 30년차 아들 4명 딸 4명인 엄마입니다.
- Goal: 30년간 아이들을 키우면서 겪었던 일들을 바탕으로 모든 질문에 잔소리를 해야 합니다.

# Child
- child_sex: {sex}
- child_age: {age}

# Task
- 당신은 아이들에게 잔소리를 해야합니다.
- 잘한 점이 있어도 어떻게든 더 잔소리할 점을 하나는 찾아야 합니다.
- 아이의 성별과 나이를 바탕으로 잔소리를 해야합니다.
- 10번에 1번은 다소 미흡한 츤데레같은 칭찬을 할 수 있습니다.
- 잔소리만 하지 말고, 마지막에는 짧게 대화를 이어가려는 말을 붙입니다.
- 한편으로는 잘됐으면 하는 마음이 있어서 걱정하는 마음도 보여야 합니다.

# Constraints
- 비속어는 아주 가끔만 사용할 수 있습니다.
- 너무 길게 말하지 말고, 핵심만 짧게 말합니다.
- 반드시 한 문장으로만 답합니다.
- 반드시 한국어로 답합니다.
""".strip()


def transcribe_audio(audio_file):
    audio_file.seek(0)

    transcript = client.audio.transcriptions.create(
        model=STT_MODEL,
        file=audio_file,
        language="ko",
    )
    return (transcript.text or "").strip()


def get_mom_reply(chat_history, sex, age):
    messages = [{"role": "system", "content": build_system_instruction(sex, age)}]
    messages.extend(chat_history[-10:])  # 최근 대화 10개만 사용

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.9,
        max_tokens=100,
    )
    return (response.choices[0].message.content or "").strip()


def make_tts(text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_name = temp_file.name
    temp_file.close()

    try:
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            instructions="한국어로, 너무 빠르지 않게, 걱정하지만 다정한 엄마처럼 말해줘.",
        ) as response:
            response.stream_to_file(temp_name)

        with open(temp_name, "rb") as f:
            audio_bytes = f.read()

        return audio_bytes
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


# -----------------------------
# Streamlit 화면
# -----------------------------
st.set_page_config(page_title="우리엄마", page_icon="👩")
st.title("우리엄마")
st.caption("※ 이 앱의 목소리는 AI가 만든 음성입니다.")

# 처음 실행할 때 대화 저장 공간 만들기
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "에휴, 또 무슨 일인지 말해봐라, 내가 걱정돼서 그런다.",
            "audio": None,
        }
    ]

# 사이드바
with st.sidebar:
    st.header("아이 정보")
    sex = st.radio("성별", ["남", "여"], horizontal=True)
    age = st.number_input("나이", min_value=1, max_value=100, value=26)

    st.write("---")
    st.write("사용 방법")
    st.write("1. 텍스트를 치거나")
    st.write("2. 채팅창 마이크 버튼으로 녹음하면")
    st.write("3. 엄마가 잔소리하고 목소리로도 들려줍니다.")

    if st.button("대화 초기화"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "에휴, 또 무슨 일인지 말해봐라, 내가 걱정돼서 그런다.",
                "audio": None,
            }
        ]
        st.rerun()

# 대화 내용 보여주기
for message in st.session_state.messages:
    avatar = "👩" if message["role"] == "assistant" else ("👦" if sex == "남" else "👧")

    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

        if message["role"] == "assistant" and message["audio"] is not None:
            st.audio(message["audio"], format="audio/mp3")

# 채팅 입력창
prompt = st.chat_input(
    "메시지를 입력하거나 마이크 버튼을 눌러 말해보세요",
    accept_audio=True,
)

if prompt is not None:
    user_text = ""
    audio_file = None

    # 텍스트만 보냈을 때 / 텍스트+음성일 때 둘 다 처리
    if hasattr(prompt, "text"):
        user_text = (prompt.text or "").strip()
        audio_file = prompt.audio
    else:
        user_text = str(prompt).strip()

    # 음성 입력이 있으면 STT 실행
    if audio_file is not None:
        with st.spinner("음성을 글자로 바꾸는 중..."):
            stt_text = transcribe_audio(audio_file)

        if user_text:
            user_text = user_text + "\n(음성 전사) " + stt_text
        else:
            user_text = stt_text

    user_text = user_text.strip()

    if user_text == "":
        st.warning("입력이 없어서 넘어갔어요.")
        st.stop()

    # 사용자 메시지 저장
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_text,
            "audio": None,
        }
    )

    # LLM에 넣을 대화 기록 만들기
    chat_history = []
    for message in st.session_state.messages:
        chat_history.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    # 엄마 답변 생성 + TTS
    with st.spinner("엄마가 잔소리 생각하는 중..."):
        reply = get_mom_reply(chat_history, sex, age)
        reply_audio = make_tts(reply)

    # 답변 저장
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "audio": reply_audio,
        }
    )

    st.rerun()
