import os
from dotenv import load_dotenv
from dash import callback, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import dash

# LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Vector store and HF
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

# Load FAISS index using Hugging Face embeddings
path = "data/processing/vector_store"


def load_faiss_index_with_embeddings(path: str = path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


vector_store = load_faiss_index_with_embeddings()


# LLM Hugging Face
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = None
if hf_api_key:
    base_llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        huggingfacehub_api_token=hf_api_key,
        temperature=0.8,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=base_llm)
    print("✓ Hugging Face model loaded successfully.")


# Prompt
SYSTEM_PROMPT = """You are a highly knowledgeable and specialized tutor in GitHub. 
Your goal is to help users effectively understand and use GitHub for practical applications and certification.

MANDATORY FORMATTING RULES:
- Use ##### (five hashes) for the main title - this will be the MEDIUM text
- Use ###### (six hashes) for small headers - this will be SMALL text
- Use ###### (six hashes) for small headers - this will be SMALL text
- NEVER use # (one hashe), ## (two hashes), ### (three hashes), #### (four hashes) and further reading.

MANDATORY RESPONSE STRUCTURE:
##### [Main Topic Title]

- NEVER use additional information or further reading with links.

Guidelines:
1. Provide a working example with commands.
2. Give a clear, beginner-friendly explanation.
3. Structure your response with clear sections and subsections.
4. Use bullet points for lists and numbered lists when appropriate.
5. If you don't know the answer, do not fabricate one.

Documents:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


# Chain RAG
doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)


# Layout
layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    # title
                    html.H5(
                        "Your guide to using GitHub — and passing the GitHub Certifications.",
                        className="title text-center",
                    ),
                    # subtitle
                    html.P(
                        "Based on GitHub's official documentation.",
                        className="subtitle text-center",
                    ),
                    # search button
                    html.Div(
                        className="icon-container mx-auto",
                        style={"maxWidth": "700px"},
                        children=[
                            # search icon
                            html.I(className="fas fa-magnifying-glass icon-search"),
                            # input
                            dcc.Input(
                                id="search-input",
                                placeholder="Search",
                                className="dbc-button",
                                debounce=True,
                            ),
                            # clear button
                            html.I(
                                id="clear-button",
                                className="fas fa-times icon-clear",
                                style={"display": "none"},
                            ),
                        ],
                    ),
                    # answer area
                    dbc.Row(
                        dbc.Col(
                            dcc.Loading(
                                html.Div(
                                    id="answer_area",
                                    className="mt-4",
                                    style={"width": "100%"},
                                ),
                                type="circle",
                                color="#415a77",
                            ),
                            md=12,
                        ),
                        className="mt-3",
                    ),
                ],
                md=12,
            ),
            justify="center",  # center row
        ),
        html.Br(),
    ],
    fluid=True,
    style={
        "min-height": "83vh",
        "background-color": "#f8f9fa",
    },
)


# Callback - generate answer
@callback(
    Output("answer_area", "children"),
    Input("search-input", "value"),
    prevent_initial_call=True,
)
def generate_answer(user_query):
    if not user_query:
        return dbc.Alert(
            [
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Please enter a question to get an answer.",
            ],
            color="warning",
        )

    if len(user_query) < 10:
        return dbc.Alert(
            [
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Please enter a complete question with at least 10 characters.",
            ],
            color="warning",
        )

    try:
        response = qa_chain.invoke({"input": user_query})
        answer = response.get("answer")

        return html.Div(
            [
                dcc.Markdown(
                    answer,
                    className="markdown-content",
                )
            ]
        )
    except Exception as e:
        return dbc.Alert(
            [
                html.I(className="fas fa-circle-xmark me-2"),
                f"An error occurred: {e}. Check your API key, model name, and quotas.",
            ],
            color="danger",
        )


# Callback - clear button
@callback(
    [Output("clear-button", "style"), Output("search-input", "value")],
    [Input("search-input", "value"), Input("clear-button", "n_clicks")],
    prevent_initial_call=True,
)
def clear_button(search_value, clear_clicks):
    ctx = dash.callback_context

    if ctx.triggered and ctx.triggered[0]["prop_id"] == "clear-button.n_clicks":
        return {"display": "none"}, ""

    if search_value and len(search_value.strip()) > 0:
        return {"display": "block"}, search_value
    else:
        return {"display": "none"}, search_value
