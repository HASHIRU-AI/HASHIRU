import os, base64
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import gradio as gr
import requests
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
def public(request: Request, user = Depends(get_user)):
    if user:
        return RedirectResponse("/gradio")
    else:
        return RedirectResponse("/main")

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

# 5. Gradio UI ---------------------------------------------------------
_logo_b64 = base64.b64encode(open("HASHIRU_LOGO.png", "rb").read()).decode()
HEADER_HTML = f"""
<div style='display:flex;align-items:center;width:30%;'>
  <img src='data:image/png;base64,{_logo_b64}' width='40' class='logo'/>
  <h1>HASHIRU AI</h1>
</div>"""

CSS = """
.logo {
    margin-right: 20px;
}
.login-status {
    font-weight: bold;
    margin-right: 20px;
    padding: 8px;
    border-radius: 4px;
    background-color: #f0f0f0;
}

/* Profile style improvements */
.profile-container {
    position: relative;
    display: inline-block;
    float: right;
    margin-right: 20px;
    z-index: 9999; /* Ensure this is higher than any other elements */
}

#profile-name {
    background-color: transparent; /* Transparent background */
    color: #f97316; /* Orange text */
    font-weight: bold;
    padding: 10px 14px;
    border-radius: 6px;
    cursor: pointer;
    user-select: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 40px;
    min-height: 40px;
    border: 2px solid #f97316; /* Add border */
}

#profile-menu {
    position: fixed; /* Changed from absolute to fixed for better overlay */
    right: auto; /* Let JS position it precisely */
    top: auto; /* Let JS position it precisely */
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 10000; /* Very high z-index to ensure it's on top */
    overflow: visible;
    width: 160px;
}

#profile-menu.hidden {
    display: none;
}

#profile-menu button {
    background-color: #f97316; /* Orange background */
    border: none;
    color: white; /* White text */
    font-size: 16px;
    border-radius: 8px;
    text-align: left;
    width: 100%;
    padding: 12px 16px;
    cursor: pointer;
    transition: background-color 0.2s ease;
    display: block;
}

#profile-menu button:hover {
    background-color: #ea580c; /* Darker orange on hover */
}

#profile-menu button .icon {
    margin-right: 8px;
    color: white; /* White icon color */
}

/* Fix dropdown issues */
input[type="text"], select {
    color: black !important;
}

/* Optional: limit dropdown scroll if options are long */
.gr-dropdown .gr-dropdown-options {
    max-height: 200px;
    overflow-y: auto;
}

/* User avatar styles */
.user-avatar {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    text-transform: uppercase;
    font-size: 20px; /* Larger font size */
    color: #f97316; /* Orange color */
}

/* Fix for gradio interface */
.gradio-container {
    overflow: visible !important;
}

/* Fix other container issues that might cause scrolling */
body, html {
    overflow-x: hidden; /* Prevent horizontal scrolling */
}

#gradio-app, .gradio-container .overflow-hidden {
    overflow: visible !important; /* Override any overflow hidden that might interfere */
}

/* Ensure dropdown appears above everything */
.profile-container * {
    z-index: 9999;
}
"""

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
    
with gr.Blocks() as login:
    btn = gr.Button("Login")
    _js_redirect = """
    () => {
        url = '/login' + window.location.search;
        window.open(url, '_blank');
    }
    """
    btn.click(None, js=_js_redirect)

app = gr.mount_gradio_app(app, login, path="/main")

with gr.Blocks(css=CSS, fill_width=True, fill_height=True) as demo:
    model_manager = GeminiManager(gemini_model="gemini-2.0-flash")

    with gr.Row():
        gr.Markdown(HEADER_HTML)

        with gr.Column(scale=1, min_width=250):
            profile_html = gr.HTML(value="""
            <div class="profile-container">
                <div id="profile-name" class="user-avatar">G</div>
                <div id="profile-menu" class="hidden">
                    <button id="login-btn" onclick="window.location.href='/login'"><span class="icon">🔐</span> Login</button>
                    <button id="logout-btn" onclick="window.location.href='/logout'"><span class="icon">🚪</span> Logout</button>
                </div>
            </div>
            """)

    with gr.Column():
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

        chatbot = gr.Chatbot(
            avatar_images=("HASHIRU_2.png", "HASHIRU.png"),
            type="messages", show_copy_button=True, editable="user",
            placeholder="Type your message here…",
        )
        gr.ChatInterface(run_model, type="messages", chatbot=chatbot, additional_outputs=[chatbot], save_history=True)

    demo.load(None, None, None, js="""
    async () => {
        const profileBtn = document.getElementById("profile-name");
        const profileMenu = document.getElementById("profile-menu");
        const loginBtn = document.getElementById("login-btn");
        const logoutBtn = document.getElementById("logout-btn");
        
        // Position menu and handle positioning
        function positionMenu() {
            const btnRect = profileBtn.getBoundingClientRect();
            profileMenu.style.position = "fixed";
            profileMenu.style.top = (btnRect.bottom + 5) + "px";
            profileMenu.style.left = (btnRect.right - profileMenu.offsetWidth) + "px"; // Align with right edge
        }
        
        // Close menu when clicking outside
        document.addEventListener('click', (event) => {
            if (!profileBtn.contains(event.target) && !profileMenu.contains(event.target)) {
                profileMenu.classList.add("hidden");
            }
        });
        
        // Toggle menu
        profileBtn.onclick = (e) => {
            e.stopPropagation();
            positionMenu(); // Position before showing
            profileMenu.classList.toggle("hidden");
            
            // If showing menu, make sure it's positioned correctly
            if (!profileMenu.classList.contains("hidden")) {
                setTimeout(positionMenu, 0); // Reposition after render
            }
        }
        
        // Handle window resize
        window.addEventListener('resize', () => {
            if (!profileMenu.classList.contains("hidden")) {
                positionMenu();
            }
        });
        
        // Get initial letter for avatar
        function getInitial(name) {
            if (name && name.length > 0) {
                return name.charAt(0);
            }
            return "?";
        }
        
        try {
            const res = await fetch('/api/login-status', { credentials: 'include' });
            const data = await res.json();
            
            if (!data.status.includes("Logged out")) {
                const name = data.status.replace("Logged in: ", "");
                profileBtn.innerHTML = `<div class="user-avatar">${getInitial(name)}</div>`;
                profileBtn.title = name;
                loginBtn.style.display = "none";
                logoutBtn.style.display = "block";
            } else {
                profileBtn.innerHTML = `<div class="user-avatar">G</div>`;
                profileBtn.title = "Guest";
                loginBtn.style.display = "block";
                logoutBtn.style.display = "none";
            }
        } catch (error) {
            console.error("Error fetching login status:", error);
            profileBtn.innerHTML = `<div class="user-avatar">?</div>`;
            profileBtn.title = "Login status unknown";
        }
    }
    """)

app = gr.mount_gradio_app(app, demo, path="/gradio",auth_dependency=get_user)

# 6. Entrypoint --------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)