import time
import logging
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from utils.llms import llm1, llm2, llm3, llm4, llm5, llm6

# Initialize logger
logger = logging.getLogger(__name__)


# Function to get API keys
def get_api_keys():
    return {"groq_api_key": "your_groq_api_key"}


# Function to execute the pipeline
def run_pipeline(theme, st_callback):
    start_time = time.time()
    logger.info("Iniciando o processamento da rota /book/generate/")

    

    # New prompt templates
    theme_selection_prompt_template = """
    Identify the central theme or concept to explore in this book.
    Theme: {theme}
    """

    theme_research_prompt_template = """
    Gather information on the theme "{theme}". Provide key resources, recent studies, or expert insights to enrich our 
    understanding of this topic.
    """

    theme_exploration_prompt_template = """
    With the gathered information on "{theme}" ({theme_research_results}), list specific aspects, subtopics, or important 
    questions related to "{theme}" that will captivate readers and add depth to our book.
    """

    key_takeaways_prompt_template = """
    Considering the exploration of "{theme}" ({theme_exploration_results}), summarize the most compelling arguments, 
    surprising facts, or crucial ideas about "{theme}" that will form the core of our book.
    """

    book_structure_prompt_template = """
    Based on the previous research and exploration of "{theme}" ({key_takeaways_results}), outline the major sections 
    or chapters. Organize them to create a logical flow and keep readers engaged.
    """

    chapter_outlines_prompt_template = """
    For each chapter recommended, derived from the book structure ({book_structure}), list the key points to cover. 
    Describe how to open the chapter to hook readers and suggest a conclusion to aim for. Do this for all chapters
    """

    alignment_chapters_prompt_template = """
    Based on the {chapter_outlines}, provide detailed guidelines for the tone, style. Include specific points to emphasize to ensure the 
    content aligns with the book's overall structure: 
        - Tone: What should be the desired tone for the chapters? (e.g., formal, informal, academic, conversational)
        - Style: Describe the writing style that fits best with the theme and target audience. Provide guidance on sentence structure, use of jargon, and level of complexity.
        - Chapter Size: Recommend an approximate word count or number of pages for each chapter.
        - Opening: Suggest how to open each chapter engagingly to hook readers.
        - Conclusion: Provide a strategy for concluding each chapter effectively.
    """

    writing_chapters_prompt_template = """
    Based on the chapter outlines ({chapter_outlines}) and considering the alignment guidelines ({alignment}), draft 
    the content for chapters 1 and 2.
    For each chapter, follow these guidelines:
        - Introduction: Provide a compelling opening that hooks the reader.
        - Main Points: Cover the key points as outlined. Ensure logical flow and coherence.
        - Tone and Style: Adhere to the specified tone and style, both suggested in the ({alignment}) phase.
        - Target Length: Aim for a chapter length suggested on the ({alignment}).
        - Conclusion: End with a strong conclusion that reinforces the main points and provides a seamless transition to the next chapter.
    """

    writing_chapters2_prompt_template = """
    Based on the chapter outlines ({chapter_outlines}) and considering the alignment guidelines ({alignment}), draft the content for chapters 3 and 4.
    For each chapter, follow these guidelines:
        - Introduction: Provide a compelling opening that hooks the reader.
        - Main Points: Cover the key points as outlined. Ensure logical flow and coherence.
        - Tone and Style: Adhere to the specified tone and style, both suggested in the ({alignment}) phase.
        - Target Length: Aim for a chapter length suggested on the ({alignment}).
        - Conclusion: End with a strong conclusion that reinforces the main points and provides a seamless transition to the next chapter.
    """

    writing_chapters3_prompt_template = """
    Based on the chapter outlines ({chapter_outlines}) and considering the alignment guidelines ({alignment}), draft the content for chapters 5 and 6.
    For each chapter, follow these guidelines:
        - Introduction: Provide a compelling opening that hooks the reader.
        - Main Points: Cover the key points as outlined. Ensure logical flow and coherence.
        - Tone and Style: Adhere to the specified tone and style, both suggested in the ({alignment}) phase.
        - Target Length: Aim for a chapter length suggested on the ({alignment}).
        - Conclusion: End with a strong conclusion that reinforces the main points and provides a seamless transition to the next chapter.
    """

    writing_chapters4_prompt_template = """
    Based on the chapter outlines ({chapter_outlines}) and considering the alignment guidelines ({alignment}), draft 
    the content for chapters 7 and 8.
    For each chapter, follow these guidelines:
        - Introduction: Provide a compelling opening that hooks the reader.
        - Main Points: Cover the key points as outlined. Ensure logical flow and coherence.
        - Tone and Style: Adhere to the specified tone and style, both suggested in the ({alignment}) phase.
        - Target Length: Aim for a chapter length suggested on the ({alignment}).
        - Conclusion: End with a strong conclusion that reinforces the main points and provides a seamless transition 
        to the next chapter.
    """

    writing_chapters5_prompt_template = """
    Based on the chapter outlines ({chapter_outlines}) and considering the alignment guidelines ({alignment}), draft 
    the content for chapters 9 and 10.
    For each chapter, follow these guidelines:
        - Introduction: Provide a compelling opening that hooks the reader.
        - Main Points: Cover the key points as outlined. Ensure logical flow and coherence.
        - Tone and Style: Adhere to the specified tone and style, both suggested in the ({alignment}) phase.
        - Target Length: Aim for a chapter length suggested on the ({alignment}).
        - Conclusion: End with a strong conclusion that reinforces the main points and provides a seamless transition 
        to the next chapter.
    """

    revision_prompt_template = """
    Considering what has been gathered on {writing_chapters}, {writing_chapters2}, {writing_chapters3}, {writing_chapters4}, and {writing_chapters5},
    present a final version of the content drafted, having in mind that that you should respect the follow guidelines:
        - Grammar and punctuation: Correct any grammatical errors and ensure proper punctuation throughout the text.
        - Spelling: Correct all spelling mistakes.
        - Tone and Style: Maintain the tone and style described in the {alignment} guidelines.
        - Cohesion and Coherence: Ensure a logical flow of ideas and smooth transitions between sections within each chapter.
        - Chapter Length: Ensure that each chapter have, at least, 15 paragraphs.
        - Engagement: Make sure that the content remains engaging and compelling for the reader.
        - Formatting: Ensure that the formatting is consistent and professional.
    """

    # Create PromptTemplates
    theme_selection_prompt = PromptTemplate(input_variables=["theme"], template=theme_selection_prompt_template)
    theme_research_prompt = PromptTemplate(input_variables=["theme"], template=theme_research_prompt_template)
    theme_exploration_prompt = PromptTemplate(input_variables=["theme", "theme_research_results"], template=theme_exploration_prompt_template)
    key_takeaways_prompt = PromptTemplate(input_variables=["theme", "theme_exploration_results"], template=key_takeaways_prompt_template)
    book_structure_prompt = PromptTemplate(input_variables=["theme", "key_takeaways_results"], template=book_structure_prompt_template)
    chapter_outlines_prompt = PromptTemplate(input_variables=["book_structure"], template=chapter_outlines_prompt_template)
    alignment_chapters_prompt = PromptTemplate(input_variables=["chapter_outlines"], template=alignment_chapters_prompt_template)
    writing_chapters_prompt = PromptTemplate(input_variables=["chapter_outlines", "alignment"], template=writing_chapters_prompt_template)
    writing_chapters2_prompt = PromptTemplate(input_variables=["chapter_outlines", "alignment"], template=writing_chapters2_prompt_template)
    writing_chapters3_prompt = PromptTemplate(input_variables=["chapter_outlines", "alignment"], template=writing_chapters3_prompt_template)
    writing_chapters4_prompt = PromptTemplate(input_variables=["chapter_outlines", "alignment"], template=writing_chapters4_prompt_template)
    writing_chapters5_prompt = PromptTemplate(input_variables=["chapter_outlines", "alignment"], template=writing_chapters5_prompt_template)
    revision_prompt = PromptTemplate(input_variables=["writing_chapters", "writing_chapters2", "writing_chapters3", "writing_chapters4", "writing_chapters5", "alignment"], template=revision_prompt_template)
    logger.info("Prompts set successfully")

    # Create chains
    theme_selection_chain = theme_selection_prompt | llm1 | StrOutputParser()
    theme_research_chain = theme_research_prompt | llm1 | StrOutputParser()
    theme_exploration_chain = theme_exploration_prompt | llm3 | StrOutputParser()
    key_takeaways_chain = key_takeaways_prompt | llm3 | StrOutputParser()
    book_structure_chain = book_structure_prompt | llm1 | StrOutputParser()
    chapter_outlines_chain = chapter_outlines_prompt | llm1 | StrOutputParser()
    alignment_chapters_chain = alignment_chapters_prompt | llm3 | StrOutputParser()
    writing_chapters_chain = writing_chapters_prompt | llm3 | StrOutputParser()
    writing_chapters2_chain = writing_chapters2_prompt | llm3 | StrOutputParser()
    writing_chapters3_chain = writing_chapters3_prompt | llm3 | StrOutputParser()
    writing_chapters4_chain = writing_chapters4_prompt | llm3 | StrOutputParser()
    writing_chapters5_chain = writing_chapters5_prompt | llm3 | StrOutputParser()
    revision_chain = revision_prompt | llm3 | StrOutputParser()

    output_theme_selection = theme_selection_chain.invoke({"theme": theme})
    st_callback.write(f"Theme Selection Output:\n{output_theme_selection}")
    logger.info("Theme selection concluded")

    output_theme_research = theme_research_chain.invoke({"theme": theme})
    st_callback.write(f"Theme Research Output:\n{output_theme_research}")
    logger.info("Theme research concluded")

    output_theme_exploration = theme_exploration_chain.invoke({"theme": theme, "theme_research_results": output_theme_research})
    st_callback.write(f"Theme Exploration Output:\n{output_theme_exploration}")
    logger.info("Theme exploration concluded")

    output_key_takeaways = key_takeaways_chain.invoke({"theme": theme, "theme_exploration_results": output_theme_exploration})
    st_callback.write(f"Key Takeaways Output:\n{output_key_takeaways}")
    logger.info("Key takeaways concluded")

    output_book_structure = book_structure_chain.invoke({"theme": theme, "key_takeaways_results": output_key_takeaways})
    st_callback.write(f"Book Structure Output:\n{output_book_structure}")
    logger.info("Book structure concluded")

    output_chapter_outlines = chapter_outlines_chain.invoke({"book_structure": output_book_structure})
    st_callback.write(f"Chapter Outlines Output:\n{output_chapter_outlines}")
    logger.info("Chapter outlines concluded")

    alignment = alignment_chapters_chain.invoke({"chapter_outlines": output_chapter_outlines})
    st_callback.write(f"Alignment Output:\n{alignment}")
    logger.info("Alignment concluded")

    output_writing_chapters = writing_chapters_chain.invoke({"chapter_outlines": output_chapter_outlines, "alignment": alignment})
    st_callback.write(f"Writing Chapters Output:\n{output_writing_chapters}")
    logger.info("Writing chapters concluded")

    output_writing_chapters2 = writing_chapters2_chain.invoke({"chapter_outlines": output_chapter_outlines, "alignment": alignment})
    st_callback.write(f"Writing Chapters 2 Output:\n{output_writing_chapters2}")
    logger.info("Writing chapters 2 concluded")

    output_writing_chapters3 = writing_chapters3_chain.invoke({"chapter_outlines": output_chapter_outlines, "alignment": alignment})
    st_callback.write(f"Writing Chapters 3 Output:\n{output_writing_chapters3}")
    logger.info("Writing chapters 3 concluded")

    output_writing_chapters4 = writing_chapters4_chain.invoke({"chapter_outlines": output_chapter_outlines, "alignment": alignment})
    st_callback.write(f"Writing Chapters 4 Output:\n{output_writing_chapters4}")
    logger.info("Writing chapters 4 concluded")

    output_writing_chapters5 = writing_chapters5_chain.invoke({"chapter_outlines": output_chapter_outlines, "alignment": alignment})
    st_callback.write(f"Writing Chapters 5 Output:\n{output_writing_chapters5}")
    logger.info("Writing chapters 5 concluded")

    revision = revision_chain.invoke({"writing_chapters": output_writing_chapters, "writing_chapters2": output_writing_chapters2, "writing_chapters3": output_writing_chapters3, "writing_chapters4": output_writing_chapters4, "writing_chapters5": output_writing_chapters5, "alignment": alignment})
    st_callback.write(f"Revision Output:\n{revision}")
    logger.info("Revision concluded")

    total_time = time.time() - start_time
    st_callback.write(f"Total Processing Time: {total_time} seconds")
    logger.info("Book generation concluded")


# Streamlit interface
st.title("Book Generator")
theme = st.text_input("Enter the book theme:")

if st.button("Generate Book"):
    with st.spinner("Generating book..."):
        st_callback = st.container()
        run_pipeline(theme, st_callback)
