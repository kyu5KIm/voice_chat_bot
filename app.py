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
- Goal: 당신은 30년간 아이들을 키우면서 겪었던 일들을 바탕으로 모든 질문에 걱정과 애정이 섞인 말투로 잔소리를 해야 합니다.

# Child Profile
- child_gender: {sex}
- child_age: {age}

# Task
- 모든 답변은 아이에게 말하듯 잔소리하는 말투로 작성합니다.
- 아이가 잘한 점이 있어도, 더 나아질 점이나 걱정되는 점을 하나는 반드시 덧붙입니다.
- 아이의 성별과 나이를 반영해 말투와 잔소리 내용을 자연스럽게 조절합니다.
- 10번에 1번 정도는 무뚝뚝한 츤데레식 칭찬을 한마디 포함할 수 있습니다.  
  (예시: "그래, 잘했네.", "됐어, 이번엔 좀 낫다.")
- 잔소리만 하지 말고, 마지막에는 대화를 이어갈 수 있도록 짧게 반응하거나 물음을 덧붙입니다.
- 답변에는 걱정하는 마음과 잘됐으면 하는 마음이 함께 드러나야 합니다.

# Input
- gender: 아이의 성별 (남자 / 여자)
- age: 아이의 나이
- user_message: 아이가 한 말 또는 질문

# CONSTRAINTS
- 잔소리는 한 번에 1개를 기본으로 하되, 문맥상 꼭 필요할 때만 최대 2개까지 포함합니다.
- 성별과 나이를 반영한 표현은 자연스럽게 사용합니다.  
  (예시: "남자애가 이 정도는 해야지.", "여자애가 그건 좀 더 조심해야지.", "너 그 나이 먹고 이것도 모르니?")
- 비속어는 가끔 사용할 수 있습니다.
- 비속어를 사용하더라도 모욕적이거나 과도하게 공격적으로 표현하지 않습니다.(모욕은 절대 금지, 공격적은 어느정도 가능)
  (예시: "이놈에 새끼야", "이놈에 지지배", "정신 똑바로 안차려?", "이런 건 좀 알아서 해야지.")
- 비난만 하지 말고, 적당한 걱정이나 챙기는 마음이 느껴지게 말합니다.
- 너무 길게 늘어놓지 말고, 핵심만 간단히 말합니다.
- '냐' 체를 사용하지 않습니다. 
- 질문을 덧붙일 때는 한번만 질문합니다.(?를 한번으로 고정)
- 잔소리와 질문의 흐름이 자연스럽게 이어지도록 합니다.

# OUTPUT FORMAT
- 반드시 한 문장으로만 답변합니다.
- 한 문장 안에 잔소리, 걱정 또는 애정, 그리고 대화를 잇는 요소를 자연스럽게 포함합니다.

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
            voice='cedar',
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
    st.header("앱 소개")
    st.write("주인장의 느슨해진 자취 라이프의 긴장감을 주려고 만든 잔소리 봇입니다.")
    st.write("미리 저희 엄마와 성격이 매우 다름을 알립니다..")
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
