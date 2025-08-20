import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from page.content import layout as page_layout


app = dash.Dash(
    __name__,
    title="GitHub Study Guide",
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
)

server = app.server

# Navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # logo -------------------------------------------------
                dbc.Row(
                    [
                        dbc.Col(
                            html.I(
                                className=("fab fa-github navbar-icon-title"),
                            ),
                        ),
                        # title -------------------------------------------------
                        dbc.Col(
                            dbc.NavbarBrand(
                                "GitHub Study Guide",
                                className="navbar-title",
                            )
                        ),
                    ],
                ),
                href="/",
                style={"textDecoration": "none"},
            ),
        ],
        fluid=True,
        className="navbar",
    ),
    # navbar style -------------------------------------------------
    color="#415a77",
    dark=True,
)

# main
content = html.Div(
    className="page-content",
)

footer = html.Footer(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                children=[
                                    html.Span("Created by", className="me-2"),
                                    html.A(
                                        "Mayara Daher",
                                        href="https://github.com/mayaradaher",
                                        target="_blank",
                                    ),
                                ],
                            )
                        ],
                        md=6,
                    ),
                ],
            )
        ],
        fluid=True,
    ),
    className="footer mt-auto bg-light border-top",
)


# font-awesome and fonts -------------------------------------------------
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <link rel="icon" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6/svgs/brands/github.svg" type="image/svg+xml">    </head>
    <body>
        {%app_entry%}
        {%config%}
        {%scripts%}
        {%renderer%}
    </body>
</html>

<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&display=swap');
</style>
"""


# layout
app.layout = html.Div(
    children=[
        dcc.Location(id="url"),
        navbar,
        html.Div(
            children=[
                content,
                html.Div(page_layout, className="page-content"),
            ],
        ),
        footer,
    ],
)


if __name__ == "__main__":
    app.run(debug=True)
