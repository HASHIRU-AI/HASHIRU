from typing import List
import gradio as gr

import base64
from src.manager.manager import GeminiManager, Mode
from enum import Enum
import os
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import requests
from src.manager.manager import GeminiManager
import argparse

# 1. Load environment --------------------------------------------------
load_dotenv()
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "replace-me")

# 2. Auth0 client ------------------------------------------------------
oauth = OAuth()
oauth.register(
    "auth0",
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f"https://{AUTH0_DOMAIN}/.well-known/openid-configuration",
)

# 3. FastAPI app -------------------------------------------------------
app = FastAPI()

# Create static directory if it doesn't exist
os.makedirs("static/fonts/ui-sans-serif", exist_ok=True)
os.makedirs("static/fonts/system-ui", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add session middleware (no auth requirement)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    session_cookie="session",
    max_age=86400,
    same_site="lax",
    https_only=False
)

# 4. Auth routes -------------------------------------------------------
# Dependency to get the current user


def get_user(request: Request):
    user = request.session.get('user')
    if user:
        return user['name']
    return None


@app.get('/')
def public(request: Request, user=Depends(get_user)):
    if user:
        return RedirectResponse("/hashiru")
    else:
        return RedirectResponse("/login-page")


@app.get("/login")
async def login(request: Request):
    print("Session cookie:", request.cookies.get("session"))
    print("Session data:", dict(request.session))
    return await oauth.auth0.authorize_redirect(request, request.url_for("auth"), audience=AUTH0_AUDIENCE, prompt="login")


@app.get("/auth")
async def auth(request: Request):
    try:
        token = await oauth.auth0.authorize_access_token(request)
        request.session["user"] = token["userinfo"]
        return RedirectResponse("/")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/logout")
async def logout(request: Request):
    auth0_logout_url = (
        f"https://{AUTH0_DOMAIN}/v2/logout"
        f"?client_id={AUTH0_CLIENT_ID}"
        f"&returnTo=http://localhost:7860/post-logout"
    )
    return RedirectResponse(auth0_logout_url)


@app.get("/post-logout")
async def post_logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")


@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "HASHIRU AI",
        "short_name": "HASHIRU",
        "icons": [],
        "start_url": "/",
        "display": "standalone"
    })


@app.get("/api/login-status")
async def api_login_status(request: Request):
    if "user" in request.session:
        user_info = request.session["user"]
        user_name = user_info.get("name", user_info.get("email", "User"))
        return {"status": f"Logged in: {user_name}"}
    else:
        return {"status": "Logged out"}


_logo_bytes = open("HASHIRU_LOGO.png", "rb").read()
_logo_b64 = base64.b64encode(_logo_bytes).decode()
_header_html = f"""
<div style="
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: flex-start;
">
  <img src="data:image/png;base64,{_logo_b64}" width="40" class="logo"/>
  <h1>
    HASHIRU AI
  </h1>
</div>
"""
css = """
.logo {
    margin-right: 20px;
}
"""


def run_model(message, history):
    if 'text' in message:
        if message['text'].strip() != "":
            history.append({
                "role": "user",
                "content": message['text']
            })
    if 'files' in message:
        for file in message['files']:
            history.append({
                "role": "user",
                "content": (file,)
            })
    yield "", history
    for messages in model_manager.run(history):
        yield "", messages


with gr.Blocks() as login:
    btn = gr.Button("Login", link="/login")

app = gr.mount_gradio_app(app, login, path="/login-page")

parser = argparse.ArgumentParser()
parser.add_argument('--no-auth', action='store_true')
args, unknown = parser.parse_known_args()
no_auth = args.no_auth

with gr.Blocks(title="HASHIRU AI", css=css, fill_width=True, fill_height=True) as demo:
    model_manager = GeminiManager(
        gemini_model="gemini-2.0-flash", modes=[mode for mode in Mode])

    def update_model(modeIndexes: List[int]):
        modes = [Mode(i+1) for i in modeIndexes]
        print(f"Selected modes: {modes}")
        model_manager.set_modes(modes)

    with gr.Column(scale=1):
        with gr.Row(scale=0):
            with gr.Column(scale=0):
                gr.Markdown(_header_html)
                gr.Button("Logout", link="/logout")
            
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=[mode.name for mode in Mode],
                    value=model_manager.get_current_modes,
                    interactive=True,
                    type="index",
                    multiselect=True,
                    label="Select Modes",
                )

                model_dropdown.change(
                    fn=update_model, inputs=model_dropdown, outputs=[])
        with gr.Row(scale=1):
            chatbot = gr.Chatbot(
                avatar_images=("HASHIRU_2.png", "HASHIRU.png"),
                type="messages",
                show_copy_button=True,
                editable="user",
                scale=1,
                render_markdown=True,
                placeholder="Type your message here...",
            )
            gr.ChatInterface(fn=run_model,
                             type="messages",
                             chatbot=chatbot,
                             additional_outputs=[chatbot],
                             save_history=True,
                             editable=True,
                             multimodal=True,)

app = gr.mount_gradio_app(app, demo, path="/hashiru", auth_dependency=get_user)

if __name__ == "__main__":
    import uvicorn

    if no_auth:
        demo.launch(favicon_path="favicon.ico", share=True, server_name="localhost")
    else:
        uvicorn.run(app, host="0.0.0.0", port=7860)
