import logging
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic


logger = logging.getLogger(__name__)

groq_api_key = "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"

anthropic_api_key = "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"

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
    model="llama3-8b-8192",
    api_key=groq_api_key,
)



llm4 = ChatAnthropic(
    temperature=0,
    model_name="claude-3-opus-20240229",
    api_key=anthropic_api_key,
    timeout=60000,
    max_retries=5,
)

llm5 = ChatAnthropic(
    temperature=0,
    model_name="claude-3-sonnet-20240229",
    api_key=anthropic_api_key,
    timeout=600,
    max_retries=5,
)

llm6 = ChatAnthropic(
    temperature=0,
    model_name="claude-3-haiku-20240307",
    api_key=anthropic_api_key,
    timeout=600,
    max_retries=5,
)