# ------------------------------ main.py ------------------------------

import os, base64
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from authlib.integrations.starlette_client import OAuth
import gradio as gr
from src.manager.manager import GeminiManager

# 1. Load environment --------------------------------------------------
load_dotenv()
AUTH0_DOMAIN        = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID     = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_AUDIENCE      = os.getenv("AUTH0_AUDIENCE")
SESSION_SECRET_KEY  = os.getenv("SESSION_SECRET_KEY", "replace‑me")

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

# 3a. *Inner* auth‑gate middleware (needs session already populated)
class RequireAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        public = ("/login", "/auth", "/logout", "/static", "/assets", "/favicon")
        if any(request.url.path.startswith(p) for p in public):
            return await call_next(request)
        if "user" not in request.session:
            return RedirectResponse("/login")
        return await call_next(request)

app.add_middleware(RequireAuthMiddleware)   # Add **first** (inner)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)  # Add **second** (outer)

# 4. Auth routes -------------------------------------------------------
@app.get("/login")
async def login(request: Request):
    return await oauth.auth0.authorize_redirect(request, request.url_for("auth"), audience=AUTH0_AUDIENCE)

@app.get("/auth")
async def auth(request: Request):
    token = await oauth.auth0.authorize_access_token(request)
    request.session["user"] = token["userinfo"]
    return RedirectResponse("/")

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(
        f"https://{AUTH0_DOMAIN}/v2/logout?client_id={AUTH0_CLIENT_ID}&returnTo=http://localhost:7860/"
    )

# 5. Gradio UI ---------------------------------------------------------
_logo_b64 = base64.b64encode(open("HASHIRU_LOGO.png", "rb").read()).decode()
HEADER_HTML = f"""
<div style='display:flex;align-items:center;width:30%;'>
  <img src='data:image/png;base64,{_logo_b64}' width='40' class='logo'/>
  <h1>HASHIRU AI</h1>
</div>"""
CSS = ".logo{margin-right:20px;}"


def run_model(message, history):
    history.append({"role": "user", "content": message})
    yield "", history
    for messages in model_manager.run(history):
        for m in messages:
            if m.get("role") == "summary":
                print("Summary:", m["content"])
        yield "", messages


def update_model(name):
    print("Model changed to:", name)


with gr.Blocks(css=CSS, fill_width=True, fill_height=True) as demo:
    model_manager = GeminiManager(gemini_model="gemini-2.0-flash")
    with gr.Column():
        with gr.Row():
            gr.Markdown(HEADER_HTML)
            model_dropdown = gr.Dropdown(
                [
                    "HASHIRU",
                    "Static-HASHIRU",
                    "Cloud-Only HASHIRU",
                    "Local-Only HASHIRU",
                    "No-Economy HASHIRU",
                ],
                value="HASHIRU",
                interactive=True,
            )
            model_dropdown.change(update_model, model_dropdown)
        with gr.Row():
            chatbot = gr.Chatbot(
                avatar_images=("HASHIRU_2.png", "HASHIRU.png"),
                type="messages", show_copy_button=True, editable="user",
                placeholder="Type your message here…",
            )
            gr.ChatInterface(run_model, type="messages", chatbot=chatbot, additional_outputs=[chatbot], save_history=True)

# Mount at root
gr.mount_gradio_app(app, demo, path="/")

# 6. Entrypoint --------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
