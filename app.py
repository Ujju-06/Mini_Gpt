import streamlit as st

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()



# LLM


llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)



# SESSION STATE


if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("💬 Chat History")

    # New Chat
    if st.button("➕ New Chat", use_container_width=True):

        st.session_state.current_chat = None

        st.rerun()

    st.divider()

    # Chat History
    for chat_name in st.session_state.chats:

        if st.button(
            chat_name,
            key=chat_name,
            use_container_width=True
        ):

            st.session_state.current_chat = chat_name

            st.rerun()

    st.divider()

    # Clear all history
    if st.button("🗑️ Clear History", use_container_width=True):

        st.session_state.chats = {}

        st.session_state.current_chat = None

        st.rerun()

st.title("Mini GPT")


# DISPLAY CURRENT CHAT


if st.session_state.current_chat is not None:

    current_chat = st.session_state.current_chat

    messages = st.session_state.chats[current_chat]

    for message in messages:

        if isinstance(message, HumanMessage):

            with st.chat_message("user"):
                st.write(message.content)

        elif isinstance(message, AIMessage):

            with st.chat_message("assistant"):
                st.write(message.content)

# USER INPUT

user_input = st.chat_input("Enter your message...")


if user_input:

    
    # Create new chat    

    if st.session_state.current_chat is None:

        chat_name = user_input[:40]

        st.session_state.chats[chat_name] = []

        st.session_state.current_chat = chat_name

    current_chat = st.session_state.current_chat


    # Save user message    

    st.session_state.chats[current_chat].append(
        HumanMessage(content=user_input)
    )


    # Show user message    

    with st.chat_message("user"):

        st.write(user_input)

    
    # Generate AI response    

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            messages = st.session_state.chats[current_chat]

            ai_response = model.invoke(messages)

        st.write(ai_response.content)
 
    # Save AI response
    

    st.session_state.chats[current_chat].append(
        AIMessage(content=ai_response.content)
    )


    # UPDATE CHAT NAME
    
    del st.session_state.chats[current_chat]


    new_chat_name = user_input[:40]

    counter = 1
    original_name = new_chat_name

    while new_chat_name in st.session_state.chats:

        new_chat_name = f"{original_name} ({counter})"

        counter += 1

# Save the chats
    st.session_state.chats[new_chat_name] = messages

    st.session_state.current_chat = new_chat_name


    # Refresh
    st.rerun()