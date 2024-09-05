import time
import logging
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from utils.llms import llm1, llm2, llm3, llm4, llm5, llm6

logger = logging.getLogger(__name__)


#TODO: insert webscraping of legal publications by OAB Number
#TODO: output the doc file on word format
#TODO: Input Management for active lawsuits



# Define Prompt Templates
new_lawsuit_draft_template = """
Using the facts, documents, and legal research, draft a complaint for a new lawsuit on the topic "{theme}".
Include the following sections:
- Introduction
- Facts of the Case
- Legal Grounds
- Demand
Facts: {facts}
Relevant Documents: {docs}
Legal Research: {research}
"""

existing_lawsuit_draft_template = """
Based on the provided information and research, draft a response to the ongoing lawsuit concerning "{theme}".
Include the following sections:
- Introduction
- Background
- Response to Claims
- Legal Grounds for Defense
- Conclusion
Facts: {facts}
Relevant Documents: {docs}
Legal Research: {research}
"""

power_of_attorney_template = "Generate a Power of Attorney document based on the given facts: {facts}"
declaration_of_poverty_template = "Generate a Declaration of Poverty document based on the given facts: {facts}"


def new_lawsuit_research_section(st_callback, theme, facts, docs):
    groq_api_key = get_api_keys()["groq_api_key"]
    draft_chain = ChatGroq(temperature=0.5, model="llama3-70b-8192", api_key=groq_api_key) | StrOutputParser()

    # Prompts
    draft_prompt = PromptTemplate(
        input_variables=["theme", "facts", "docs", "research"],
        template=new_lawsuit_draft_template
    )

    research_prompt = DuckDuckGoSearchRun() | StrOutputParser()

    # Research articles and books
    search_articles_output = research_prompt.invoke({"query": f"{theme} legal articles and books"})
    st_callback.write(f"Pesquisas de Artigos e Livros:\n{search_articles_output}")
    logger.info("Pesquisa de artigos e livros concluída")

    # Research previous decisions
    search_decisions_output = research_prompt.invoke({"query": f"{theme} previous legal decisions"})
    st_callback.write(f"Pesquisa de Decisões Anteriores:\n{search_decisions_output}")
    logger.info("Pesquisa de decisões anteriores concluída")

    # Process uploaded documents
    doc_names = [doc.name for doc in docs] if docs else []
    doc_info = ", ".join(doc_names) if doc_names else "No documents uploaded"

    # First Draft of the Lawsuit Petition
    draft_output = draft_chain.invoke(draft_prompt.format(
        theme=theme,
        facts=facts,
        docs=doc_info,
        research=f"{search_articles_output}\n{search_decisions_output}"
    ))
    st_callback.write(f"Primeiro Rascunho da Petição:\n{draft_output}")
    logger.info("Primeiro rascunho da petição concluído")

    return draft_output

def existing_lawsuit_research_section(st_callback, theme, facts, docs):
    groq_api_key = get_api_keys()["groq_api_key"]
    draft_chain = ChatGroq(temperature=0.5, model="llama3-70b-8192", api_key=groq_api_key) | StrOutputParser()

    # Prompts
    draft_prompt = PromptTemplate(
        input_variables=["theme", "facts", "docs", "research"],
        template=existing_lawsuit_draft_template
    )

    research_prompt = DuckDuckGoSearchRun() | StrOutputParser()

    # Research books and articles
    search_articles_output = research_prompt.invoke({"query": f"{theme} legal articles and books"})
    st_callback.write(f"Pesquisas de Artigos e Livros:\n{search_articles_output}")
    logger.info("Pesquisa de artigos e livros concluída")

    # Research previous decisions about the case
    search_decisions_output = research_prompt.invoke({"query": f"{theme} previous legal decisions"})
    st_callback.write(f"Pesquisa de Decisões Anteriores:\n{search_decisions_output}")
    logger.info("Pesquisa de decisões anteriores concluída")

    # First Draft of the Petition
    draft_output = draft_chain.invoke(draft_prompt.format({
        "theme": theme,
        "facts": facts,
        "docs": docs,
        "research": f"{search_articles_output}\n{search_decisions_output}"
    }))
    st_callback.write(f"Primeiro Rascunho da Petição:\n{draft_output}")
    logger.info("Primeiro rascunho da petição concluído")

    return draft_output


def nova_acao_judicial():
    st.title("Seção de Nova Ação Judicial")
    st.subheader("Área dos Advogados")
    areas_advogados = ["Direito Civil", "Direito Imobiliário", "Direito Trabalhista", "Direito Familiar",
                       "Direito Criminal"]
    advogado_selecionado = st.selectbox("Selecione a área de atuação do advogado", areas_advogados)

    st.subheader("Tipo da Ação Judicial")
    tipo_acao = st.text_input("Especifique o tipo da ação judicial")

    st.subheader("Documentação Necessária")
    docs_necessarios = st.file_uploader("Envie os documentos necessários", accept_multiple_files=True)

    st.subheader("Fatos do Cliente")
    fatos = st.text_area("Colete os fatos do caso")

    st.subheader("Tribunal de Jurisdição")
    tribunal_jurisdicional = st.text_input("Especifique a qual tribunal a ação judicial será submetida")

    st.subheader("Modelos de Petições para Redação e Manutenção de Estilo")
    modelos_peticoes = st.file_uploader("Envie modelos de petições", accept_multiple_files=True)

    gerar_po_btn = st.button("Gerar Procuração")
    gerar_decl_pobreza_btn = st.button("Gerar Declaração de Pobreza")

    if st.button("Gerar Nova Ação Judicial"):
        if not tipo_acao:
            st.error("Por favor, especifique o tipo da ação judicial.")
        elif not fatos:
            st.error("Por favor, forneça os fatos do caso.")
        else:
            with st.spinner("Gerando pesquisa e petição..."):
                st_callback = st.container()
                draft_output = new_lawsuit_research_section(st_callback, tipo_acao, fatos, docs_necessarios)
                st_callback.write(f"Pesquisa Completa e Primeiro Rascunho: {draft_output}")

        if gerar_po_btn:
            poa_chain = ChatGroq(temperature=0.5, model="llama3-70b-8192",
                                 api_key=get_api_keys()["groq_api_key"]) | StrOutputParser()
            poa_prompt = PromptTemplate(input_variables=["facts"], template=power_of_attorney_template)
            poa_output = poa_chain.invoke(poa_prompt.format(facts=fatos))
            st.write(f"Procuração Gerada:\n{poa_output}")

        if gerar_decl_pobreza_btn:
            poverty_chain = ChatGroq(temperature=0.5, model="llama3-70b-8192",
                                     api_key=get_api_keys()["groq_api_key"]) | StrOutputParser()
            poverty_prompt = PromptTemplate(input_variables=["facts"], template=declaration_of_poverty_template)
            poverty_output = poverty_chain.invoke(poverty_prompt.format(facts=fatos))
            st.write(f"Declaração de Pobreza Gerada:\n{poverty_output}")

def acao_judicial_existente():
    st.title("Seção de Ação Judicial Existente")
    st.subheader("Área dos Advogados")
    areas_advogados = [
        "Direito Civil", 
        "Direito Imobiliário", 
        "Direito Trabalhista", 
        "Direito Familiar", 
        "Direito Criminal", 
        "Direito Administrativo",
        "Direito Tributário",
        "Direito Internacional",
        "Direito Processual Civil",
        "Direito Processual Criminal",
        "Direito Processual Administrativo",
        "Direito Processual Tributário",
        "Direito Processual Internacional",
        ]
    area_atuacao = st.selectbox("Selecione a área de atuação do advogado", areas_advogados)

    st.subheader("Quem Está Sendo Defendido")
    cliente = st.text_input("Informe quem está sendo defendido")

    st.subheader("Expectativa da Petição")
    expectativa = st.text_area("Descreva o que você espera que a AILawyer produza nesta petição")

    st.subheader("Em Qual Fase a Petição está sendo redigida")
    fases = ["Inicial", "Conhecimento", "Provas", "Apelação", "Recurso Especial", "Recurso Extraordinário"]
    fase_selecionada = st.radio("Selecione a fase", fases)

    st.subheader("Qual é o Seu Objetivo")
    objetivo = st.text_input("Especifique o objetivo")

    st.subheader("Decisão que Você Está Respondendo")
    doc_decisao = st.file_uploader("Envie o documento da decisão")

    st.subheader("Documentos para Contexto")
    docs_contexto = st.file_uploader("Envie a petição inicial e a resposta", accept_multiple_files=True)

    st.subheader("Leis Relacionadas ao Caso")
    leis = st.text_area("Especifique as leis relevantes")

    st.subheader("Modelos de Petições para Redação e Manutenção de Estilo")
    habilitar_peticoes = st.checkbox("Habilitar envio de modelos de petições")
    if habilitar_peticoes:
        modelos_peticoes = st.file_uploader("Envie modelos de petições", accept_multiple_files=True)

    st.subheader("Documentos Anexados à Petição")
    habilitar_docs = st.checkbox("Habilitar envio de documentos anexados à petição")
    if habilitar_docs:
        docs_anexados = st.file_uploader("Envie documentos anexados à petição", accept_multiple_files=True)

    if st.button("Gerar Pesquisa e Petição"):
        with st.spinner("Gerando pesquisa e petição..."):
            st_callback = st.container()
            draft_output = existing_lawsuit_research_section(st_callback, objetivo, leis, docs_contexto)
            st_callback.write(f"Pesquisa Completa e Primeiro Rascunho: {draft_output}")


def main():
    st.sidebar.title("Navegação")
    pagina = st.sidebar.radio("Ir para", ["Nova Ação Judicial", "Ação Judicial Existente"])

    if pagina == "Nova Ação Judicial":
        nova_acao_judicial()
    elif pagina == "Ação Judicial Existente":
        acao_judicial_existente()


if __name__ == "__main__":
    main()