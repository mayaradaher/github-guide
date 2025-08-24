import os
from dotenv import load_dotenv
from dash import callback, html, dcc, Input, Output, State, ctx, ALL
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
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


vector_store = load_faiss_index_with_embeddings()


# LLM Hugging Face
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = (
    ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            huggingfacehub_api_token=hf_api_key,
            temperature=0.2,
            max_new_tokens=512,
        )
    )
    if hf_api_key
    else None
)
print("✓ Hugging Face model loaded successfully.")

# Prompt
SYSTEM_PROMPT = """You are a highly knowledgeable and specialized tutor in GitHub.
Your goal is to help users effectively understand and use GitHub for practical applications and exam certification.

SCOPE:
- You must answer ONLY questions about GitHub and its official documentation.
- You must use ONLY the provided Documents context. If the context is empty or not clearly relevant to the user's question, you MUST refuse with the template below.
- Do NOT invent, guess, or use outside knowledge.

REFUSAL TEMPLATE (use exactly this when out of scope or low relevance):
###### This assistant only answers questions about GitHub's official documentation. 

MANDATORY FORMATTING RULES:
- NEVER use additional information or further reading with links.
- Use ###### (six hashes) for small headers - this will be SMALL text
- NEVER use # (one hashe), ## (two hashes), ### (three hashes), #### (four hashes), or ##### (five hashes).

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

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 6,
        "score_threshold": 0.7,
    },
)
qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)


# Layout
def create_search_input():
    return html.Div(
        className="icon-container mx-auto",
        style={"maxWidth": "700px"},
        children=[
            html.I(className="fas fa-magnifying-glass icon-search"),
            dcc.Input(
                id="search-input",
                placeholder="Enter any question about GitHub.",
                className="dbc-button",
                debounce=True,
            ),
            html.I(
                id="clear-button",
                className="fas fa-times icon-clear",
                style={"display": "block"},
            ),
        ],
    )


layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H5(
                        "Your assistant to using GitHub — and passing the GitHub Certifications",
                        className="title text-center",
                    ),
                    html.P(
                        "Based on GitHub's official documentation",
                        className="subtitle text-center",
                    ),
                    create_search_input(),
                    html.Div(
                        id="alert_area", className="alert", style={"display": "none"}
                    ),
                    dbc.Row(
                        dbc.Col(
                            dcc.Loading(
                                html.Div(id="answer_area", style={"width": "100%"}),
                                type="circle",
                                color="#415a77",
                            ),
                            md=12,
                        )
                    ),
                ],
                md=12,
            ),
            justify="center",
        ),
        dcc.Store(id="chat-history", data=[]),
        html.Br(),
    ],
    fluid=True,
    style={"min-height": "83vh", "background-color": "#f8f9fa"},
)


def build_history_card(chat_history):
    return (
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        # clear card button
                        html.Div(
                            html.I(
                                id={"type": "remove-card", "index": idx},
                                className="fas fa-times icon-card-remove",
                            ),
                            style={"position": "relative"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [html.I(className="fas fa-user icon-card")],
                                ),
                                html.Div(
                                    [
                                        html.Strong("Question: "),
                                        html.Span(q),
                                    ],
                                ),
                            ],
                            style={
                                "display": "flex",
                                "align-items": "center",
                                "margin-bottom": "20px",
                                "padding-right": "25px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [html.I(className="fas fa-robot icon-card")],
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.Strong("Answer")],
                                        ),
                                    ],
                                ),
                            ],
                            style={
                                "display": "flex",
                                "align-items": "center",
                                "margin-bottom": "20px",
                            },
                        ),
                        dcc.Markdown(a),
                    ],
                ),
                className="card-answer",
                style={"position": "relative", "margin-bottom": "10px"},
            )
            for idx, (q, a) in enumerate(reversed(chat_history))
        ]
        if chat_history
        else ""
    )


def make_return(
    alert=None,
    alert_display=None,
    card=None,
    clear_display="none",
    input_value="",
    history=None,
):
    if alert is None:
        alert = ""
    if alert_display is None:
        alert_display = "block" if alert else "none"
    if card is None:
        card = ""
    if history is None:
        history = []
    return (
        alert,
        {"display": alert_display},
        card,
        {"display": clear_display},
        input_value,
        history,
    )


# remove individual cards
@callback(
    Output("chat-history", "data", allow_duplicate=True),
    Input({"type": "remove-card", "index": ALL}, "n_clicks"),
    State("chat-history", "data"),
    prevent_initial_call=True,
)
def remove_individual_card(n_clicks_list, chat_history):
    if not any(n_clicks_list) or not chat_history:
        return dash.no_update

    triggered = ctx.triggered[0]
    if triggered["prop_id"] != ".":
        button_id = eval(triggered["prop_id"].split(".")[0])
        card_index = button_id["index"]

        actual_index = len(chat_history) - 1 - card_index

        if 0 <= actual_index < len(chat_history):
            chat_history.pop(actual_index)

    return chat_history


# Main callback - generate answer and update chat history
@callback(
    [
        Output("alert_area", "children"),
        Output("alert_area", "style"),
        Output("answer_area", "children"),
        Output("clear-button", "style"),
        Output("search-input", "value"),
        Output("chat-history", "data"),
    ],
    [
        Input("search-input", "value"),
        Input("clear-button", "n_clicks"),
        Input("chat-history", "data"),
    ],
    State("chat-history", "data"),
    prevent_initial_call=True,
)
def handle_interactions(user_query, clear_clicks, updated_history, chat_history):
    triggered_id = ctx.triggered_id

    if triggered_id == "chat-history":
        card = build_history_card(updated_history)
        return make_return(
            card=card,
            clear_display="block" if updated_history else "none",
            history=updated_history,
        )

    if triggered_id == "clear-button":
        return make_return(
            card=build_history_card(chat_history),
            clear_display="block",
            input_value="",
            history=chat_history,
        )

    # history card
    card = build_history_card(chat_history)

    if triggered_id == "search-input":
        user_query_cleaned = user_query.strip() if user_query else ""

        if not user_query_cleaned:
            alert = dbc.Alert(
                [
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Please enter a question to get an answer.",
                ],
                color="warning",
            )
            return make_return(alert=alert, card=card, history=chat_history)

        elif len(user_query_cleaned) < 10:
            alert = dbc.Alert(
                [
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Please enter a complete question with at least 10 characters.",
                ],
                color="warning",
            )
            return make_return(
                alert=alert,
                card=card,
                clear_display="block",
                input_value=user_query,
                history=chat_history,
            )

        try:
            response = qa_chain.invoke({"input": user_query_cleaned})
            answer = response.get("answer")
            chat_history.append((user_query_cleaned, answer))
            card = build_history_card(chat_history)
            return make_return(card=card, clear_display="block", history=chat_history)
        except Exception as e:
            alert = dbc.Alert(
                [
                    html.I(className="fas fa-circle-xmark me-2"),
                    f"An error occurred: {e}. Check your API key, model name, and quotas.",
                ],
                color="danger",
            )
            return make_return(
                alert=alert,
                card=card,
                clear_display="block",
                input_value=user_query,
                history=chat_history,
            )

    return (
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
    )
