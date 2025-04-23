import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import random
import time

from agent import refinement_loop, send_query_to_agent

# Set page config
st.set_page_config(page_title="Image Chatbot", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# Function to generate smaller mock image responses
def generate_image_response(prompt):
    # Reduced image size (about 3x smaller than original)
    img_links = None

    try:
        img_links = send_query_to_agent(refinement_loop, prompt)
    except:
        img_links = send_query_to_agent(refinement_loop, prompt)

    img_list = []
    print('image links - ', img_links)
    for link in list(img_links):
        with open(link, 'rb') as f:
            img_list.append(f.read())

    return img_list

# Main app layout
st.title("Image Chatbot")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            if message.get("type") == "images":
                # Display images vertically
                for img_bytes in message["content"]:
                    st.image(img_bytes, width=300)  # Fixed width

# Input form at bottom with compact layout
with st.form(key="input_form", clear_on_submit=True):
    # Create columns with more space for input (15:1 ratio)
    input_col, button_col = st.columns([15, 1])
    
    with input_col:
        user_input = st.text_input(
            "Ask a question", 
            key="user_input",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
    
    with button_col:
        # Narrower send button
        st.write("")  # Vertical spacer
        submit_button = st.form_submit_button("➤", help="Send message")  # Arrow symbol

# Process input
if submit_button and user_input and not st.session_state.processing:
    st.session_state.processing = True
    
    # Add user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate and add assistant response
    with st.spinner("Generating response..."):
        image_responses = generate_image_response(user_input)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": image_responses,
            "type": "images"
        })
    
    st.session_state.processing = False
    st.rerun()

# Custom CSS for layout
st.markdown("""
    <style>
        /* Fixed input area at bottom */
        .main > div {
            padding-bottom: 3rem;
        }
        .stForm {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 0.5rem 1rem;
            background-color: white;
            z-index: 100;
            border-top: 1px solid #e6e6e6;
        }
        /* Compact send button */
        .stButton button {
            min-width: 40px !important;
            width: 40px !important;
            height: 38px !important;
            padding: 0 !important;
            margin-top: 1px;
        }
        /* Chat message styling */
        .stChatMessage {
            max-width: 85%;
        }
        /* Image styling */
        .stImage img {
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)
