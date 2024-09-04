import time
import logging
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from utils.llms import llm1, llm2, llm3, llm4, llm5, llm6

# Initialize logger
logger = logging.getLogger(__name__)


# Function to execute the pipeline
def run_pipeline(theme):
    start_time = time.time()
    logger.info("Iniciando o processamento da rota /podcast/generate/")

    # Define all prompts based on the content generation process
    planning_prompt_template = """Provide a detailed overview about the theme: {theme}. Consider the main aspects, 
    subtopics, and key points that should be covered. Think about the relevance and importance of each aspect to the
     overall theme."""

    research_articles_prompt_template = """Research and list the most relevant and high-quality articles and books 
    related to the theme: {theme}. Consider the credibility of the sources, the depth of information provided, and 
    how well they align with the main aspects and subtopics identified in the planning stage."""

    research_websites_prompt_template = """Research and list the most informative and trustworthy websites related to 
    the theme: {theme}. Evaluate the websites based on their authority, accuracy, and how well they complement the 
    information gathered from articles and books."""

    target_public_analysis_prompt_template = """Analyze the target public for the theme: {theme}. Consider their
     demographics, interests, knowledge level, and potential questions or concerns they may have. Think about how 
     the content should be tailored to effectively communicate with this specific audience."""

    language_adaptation_prompt_template = """Adapt the language of the following content for the target public: 
    {content}. Consider the target public's demographics, interests, and knowledge level. Think about how 
    to simplify complex concepts, use relatable examples, and maintain an engaging tone that resonates
     with the audience."""

    first_draft_prompt_template = """Create the first revised main draft based on the content: {content}. Consider 
    the information gathered during the research phase and the insights from the target public analysis. Think about
    how to organize the content logically, ensure a smooth flow of ideas, and effectively convey the main points of
    the theme."""

    keynote_prompt_template = """Generate text for a keynote presentation based on the theme: {theme}. Consider the 
    main aspects and key points identified in the planning stage. Think about how to present the information in a 
    clear, concise, and engaging manner suitable for a live presentation. Use storytelling techniques and examples
     to make the content memorable and impactful."""

    linkedin_prompt_template = """Generate an article for LinkedIn based on the theme: {theme}. Consider the target 
    audience on LinkedIn, which typically consists of professionals and industry experts. Think about how to present 
    the information in a way that demonstrates thought leadership, provides valuable insights, and encourages 
    engagement and discussion. Use a professional tone and include relevant examples and data to support 
    your points."""

    twitter_prompt_template = """Generate a series of concise and engaging tweets based on the theme: {theme}. Consider
     the character limit on Twitter and the fast-paced nature of the platform. Think about how to break down the main 
     points into bite-sized pieces of information that can be easily understood and shared. Use hashtags, mentions, 
     and compelling visuals to increase visibility and engagement. Encourage retweets and replies by asking questions 
     or making thought-provoking statements."""

    # Create prompt templates
    planning_prompt = PromptTemplate(input_variables=["theme"], template=planning_prompt_template)
    research_articles_prompt = PromptTemplate(input_variables=["theme"], template=research_articles_prompt_template)
    research_websites_prompt = PromptTemplate(input_variables=["theme"], template=research_websites_prompt_template)
    target_public_analysis_prompt = PromptTemplate(input_variables=["theme"],
                                                   template=target_public_analysis_prompt_template)
    language_adaptation_prompt = PromptTemplate(input_variables=["content"],
                                                template=language_adaptation_prompt_template)
    first_draft_prompt = PromptTemplate(input_variables=["content"], template=first_draft_prompt_template)
    keynote_prompt = PromptTemplate(input_variables=["theme"], template=keynote_prompt_template)
    linkedin_prompt = PromptTemplate(input_variables=["theme"], template=linkedin_prompt_template)
    twitter_prompt = PromptTemplate(input_variables=["theme"], template=twitter_prompt_template)

    # Define chains for each stage
    planning_chain = planning_prompt | llm1 | StrOutputParser()
    search_tool = DuckDuckGoSearchRun()

    # Web search for research articles and books
    research_articles_results = search_tool.run(f"articles and books about {theme}")
    research_articles_chain = research_articles_prompt | llm2 | StrOutputParser()
    research_articles_output = research_articles_chain.invoke({"theme": research_articles_results})

    # Web search for research websites
    research_websites_results = search_tool.run(f"websites about {theme}")
    research_websites_chain = research_websites_prompt | llm3 | StrOutputParser()
    research_websites_output = research_websites_chain.invoke({"theme": research_websites_results})

    target_public_analysis_chain = target_public_analysis_prompt | llm1 | StrOutputParser()
    language_adaptation_chain = language_adaptation_prompt | llm2 | StrOutputParser()
    first_draft_chain = first_draft_prompt | llm3 | StrOutputParser()
    keynote_chain = keynote_prompt | llm1 | StrOutputParser()
    linkedin_chain = linkedin_prompt | llm2 | StrOutputParser()
    twitter_chain = twitter_prompt | llm3 | StrOutputParser()

    # Run the chains and display the output
    st.write("### Planning Stage")
    planning_output = planning_chain.invoke({"theme": theme})
    st.write(f"Planning Output:\n{planning_output}")

    st.write("### Research Stage")
    st.write(f"Research Articles Results:\n{research_articles_results}")
    st.write(f"Research Articles Output:\n{research_articles_output}")

    st.write(f"Research Websites Results:\n{research_websites_results}")
    st.write(f"Research Websites Output:\n{research_websites_output}")

    st.write("### Raw Content Stage")
    target_public_analysis_output = target_public_analysis_chain.invoke({"theme": theme})
    st.write(f"Target Public Analysis Output:\n{target_public_analysis_output}")

    language_adaptation_output = language_adaptation_chain.invoke({"content": research_articles_output})
    st.write(f"Language Adaptation Output:\n{language_adaptation_output}")

    first_draft_output = first_draft_chain.invoke({"content": language_adaptation_output})
    st.write(f"First Revised Main Draft Output:\n{first_draft_output}")

    st.write("### Content Generation Stage")
    keynote_output = keynote_chain.invoke({"theme": theme})
    st.write(f"Keynote Presentation Output:\n{keynote_output}")

    linkedin_output = linkedin_chain.invoke({"theme": theme})
    st.write(f"LinkedIn Article Output:\n{linkedin_output}")

    twitter_output = twitter_chain.invoke({"theme": theme})
    st.write(f"Twitter Posts Output:\n{twitter_output}")

    total_time = time.time() - start_time
    st.write(f"Total Processing Time: {total_time} seconds")
    logger.info("Processamento total concluído")


# Streamlit interface
st.title("Content Generation Pipeline")
theme = st.text_input("Enter the content theme:")

if st.button("Generate Content"):
    with st.spinner("Generating content..."):
        run_pipeline(theme)