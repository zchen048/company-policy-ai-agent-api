from fastapi import FastAPI
from database import init_db
from routes import user_routes, chat_routes, message_routes

app = FastAPI()

# Create DB tables if needed
@app.on_event("startup")
def on_startup():
    init_db()

# Include routes
app.include_router(user_routes.router)
app.include_router(chat_routes.router)
app.include_router(message_routes.router)