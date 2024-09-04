import streamlit as st
from langchain_anthropic import ChatAnthropic, Anthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os

# Set up Anthropic API key
def get_api_keys():
    return {"anthropic_api_key": "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"}
# Initialize Langchain with Anthropic's LLM
llm = Anthropic(temperature=0.7)
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    api_key=anthropic_api_key
)

# Streamlit app
st.set_page_config(page_title="AI Module Idea Explorer", layout="wide")

# Initialize session state
if 'phase' not in st.session_state:
    st.session_state.phase = "home"
if 'ideas' not in st.session_state:
    st.session_state.ideas = []
if 'plan' not in st.session_state:
    st.session_state.plan = ""
if 'prd' not in st.session_state:
    st.session_state.prd = ""
if 'prototype' not in st.session_state:
    st.session_state.prototype = ""


# Home screen
def home_screen():
    st.title("AI Module Idea Explorer")
    if st.button("Start Brainstorming"):
        st.session_state.phase = "brainstorming"
        st.rerun()


# Brainstorming phase
def brainstorming_phase():
    st.title("Brainstorming Phase")

    # Sidebar
    with st.sidebar:
        st.subheader("Session Info")
        st.write(f"Ideas generated: {len(st.session_state.ideas)}")

    # Main chat area
    for idea in st.session_state.ideas:
        st.text(f"You: {idea}")
        with st.spinner("AI is thinking..."):
            response = conversation.predict(input=f"Provide feedback and suggestions for this AI project idea: {idea}")
        st.text(f"AI: {response}")

    # Input field
    new_idea = st.text_input("Enter your AI project idea:")
    if st.button("Send"):
        st.session_state.ideas.append(new_idea)
        st.rerun()

    if st.button("Finalize Brainstorming"):
        st.session_state.phase = "planning"
        st.rerun()


# Planning phase
def planning_phase():
    st.title("Planning Phase")

    # Sidebar
    with st.sidebar:
        st.subheader("Brainstorming Summary")
        for idea in st.session_state.ideas:
            st.write(f"- {idea}")

    # Main planning area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actionable Steps")
        steps = st.text_area("Enter actionable steps:", height=200)

    with col2:
        st.subheader("Milestones")
        milestones = st.text_area("Enter project milestones:", height=200)

    if st.button("Generate Plan"):
        prompt = f"Generate a comprehensive plan based on these ideas: {st.session_state.ideas}, steps: {steps}, and milestones: {milestones}"
        with st.spinner("Generating plan..."):
            st.session_state.plan = conversation.predict(input=prompt)
        st.rerun()

    if st.session_state.plan:
        st.subheader("Generated Plan")
        st.write(st.session_state.plan)

    if st.button("Finalize Planning"):
        st.session_state.phase = "output"
        st.rerun()


# Output phase
def output_phase():
    st.title("Output")

    tab1, tab2 = st.tabs(["Product Requirements Document", "Prototype Example"])

    with tab1:
        if not st.session_state.prd:
            with st.spinner("Generating PRD..."):
                prd_prompt = f"Generate a detailed Product Requirements Document based on these ideas: {st.session_state.ideas} and this plan: {st.session_state.plan}"
                st.session_state.prd = conversation.predict(input=prd_prompt)
        st.write(st.session_state.prd)
        st.download_button("Download PRD", st.session_state.prd, "product_requirements_document.txt")

    with tab2:
        if not st.session_state.prototype:
            with st.spinner("Generating Prototype Example..."):
                prototype_prompt = f"Generate a prototype example description based on these ideas: {st.session_state.ideas} and this plan: {st.session_state.plan}"
                st.session_state.prototype = conversation.predict(input=prototype_prompt)
        st.write(st.session_state.prototype)
        st.download_button("Download Prototype Example", st.session_state.prototype, "prototype_example.txt")

    if st.button("Start New Project"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# Main app logic
def main():
    if st.session_state.phase == "home":
        home_screen()
    elif st.session_state.phase == "brainstorming":
        brainstorming_phase()
    elif st.session_state.phase == "planning":
        planning_phase()
    elif st.session_state.phase == "output":
        output_phase()


if __name__ == "__main__":
    main()