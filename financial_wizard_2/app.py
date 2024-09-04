import time
import logging
import streamlit as st
from  langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchResults


# Initialize logger
logger = logging.getLogger(__name__)

# Function to get API keys
def get_api_keys():
    return {"groq_api_key": "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"}

# Function to execute the pipeline
def run_pipeline(company):
    start_time = time.time()
    logger.info("Iniciando o processamento da rota /stock_analysis/")

    api_keys = get_api_keys()
    groq_api_key = api_keys["groq_api_key"]

    if not groq_api_key:
        logger.error("Groq API key está ausente")
        raise Exception("Groq API key is missing")

    llm1 = ChatGroq(
        temperature=0.5,
        model="llama3-70b-8192",
        api_key=groq_api_key,
    )

    llm2 = ChatGroq(
        temperature=0.5,
        model="gemma2-9b-it",
        api_key=groq_api_key,
    )

    llm3 = ChatGroq(
        temperature=0.5,
        model="llama3-groq-8b-8192-tool-use-preview",
        api_key=groq_api_key,
    )

    # Define prompts
    history_prompt_template = "Research the history of the company: {company}."
    news_prompt_template = "Research the news of the last month for the company: {company}."
    financial_results_prompt_template = "Research the latest financial results for the company: {company}."
    kpis_prompt_template = "Analyze the key performance indicators (KPIs) for the company: {company}."
    technical_analysis_prompt_template = "Perform a technical analysis for the company: {company}."
    value_investing_kpis_prompt_template = "Evaluate the value investing KPIs for the company: {company}."
    report_consolidation_prompt_template = "Consolidate the following information into a comprehensive report: {information}."

    # Create prompt templates
    history_prompt = PromptTemplate(input_variables=["company"], template=history_prompt_template)
    news_prompt = PromptTemplate(input_variables=["company"], template=news_prompt_template)
    financial_results_prompt = PromptTemplate(input_variables=["company"], template=financial_results_prompt_template)
    kpis_prompt = PromptTemplate(input_variables=["company"], template=kpis_prompt_template)
    technical_analysis_prompt = PromptTemplate(input_variables=["company"], template=technical_analysis_prompt_template)
    value_investing_kpis_prompt = PromptTemplate(input_variables=["company"], template=value_investing_kpis_prompt_template)
    report_consolidation_prompt = PromptTemplate(input_variables=["information"], template=report_consolidation_prompt_template)

    # Define chains for each stage
    history_chain = history_prompt | llm1 | StrOutputParser()
    news_chain = news_prompt | llm2 | StrOutputParser()
    financial_results_chain = financial_results_prompt | llm3 | StrOutputParser()
    kpis_chain = kpis_prompt | llm1 | StrOutputParser()
    technical_analysis_chain = technical_analysis_prompt | llm2 | StrOutputParser()
    value_investing_kpis_chain = value_investing_kpis_prompt | llm3 | StrOutputParser()
    report_consolidation_chain = report_consolidation_prompt | llm1 | StrOutputParser()

    # Web search for research stages
    search_tool = DuckDuckGoSearchResults()

    st.write("### Planning Stage")
    history_output = history_chain.invoke({"company": company})
    st.write(f"History Output:\n{history_output}")

    news_results = search_tool.run(f"{company} news last month")
    news_output = news_chain.invoke({"company": news_results})
    st.write(f"News Results:\n{news_results}")
    st.write(f"News Output:\n{news_output}")

    financial_results_results = search_tool.run(f"{company} latest financial results for the last 5 years")
    financial_results_output = financial_results_chain.invoke({"company": financial_results_results})
    st.write(f"Financial Results:\n{financial_results_results}")
    st.write(f"Financial Results Output:\n{financial_results_output}")

    st.write("### Analysis Stage")
    kpis_output = kpis_chain.invoke({"company": company})
    st.write(f"KPIs Output:\n{kpis_output}")

    technical_analysis_output = technical_analysis_chain.invoke({"company": company})
    st.write(f"Technical Analysis Output:\n{technical_analysis_output}")

    value_investing_kpis_output = value_investing_kpis_chain.invoke({"company": company})
    st.write(f"Value Investing KPIs Output:\n{value_investing_kpis_output}")

    report_consolidation_output = report_consolidation_chain.invoke({"information": f"{kpis_output}\n{technical_analysis_output}\n{value_investing_kpis_output}"})
    st.write(f"Report Consolidation Output:\n{report_consolidation_output}")

    total_time = time.time() - start_time
    st.write(f"Total Processing Time: {total_time} seconds")
    logger.info("Processamento total concluído")

# Streamlit interface
st.title("Stock Analysis Pipeline")
company = st.text_input("Enter the company name:")

if st.button("Generate Stock Analysis"):
    with st.spinner("Generating stock analysis..."):
        run_pipeline(company)
