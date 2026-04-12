import streamlit as st
from chat_modal import chatbot
from langchain_core.messages import HumanMessage

st.title("LangGraph Chatbot")

if "message_history" not in st.session_state:
    st.session_state.message_history = []

# Show chat history
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
user_input = st.chat_input("Type here")

if user_input:

    # Save user message
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
           message_chunk.content 
           for message_chunk , metadata in chatbot.stream(
                 {"messages": [HumanMessage(content=user_input)]},
                 config = {"configurable": {"thread_id": "thread-1"}},
                 stream_mode = 'messages'
                 
            )
        )

        # Save assistant message
    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_message}
    )
