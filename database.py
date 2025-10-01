from sqlmodel import SQLModel, create_engine, Session

DATABASE_URL = "sqlite:///app.db"  

# Create engine
engine = create_engine(DATABASE_URL, echo=True)  # echo=True logs SQL

# Create tables function
def init_db():
    SQLModel.metadata.create_all(engine)

# Dependency to get a DB session in routes
def get_session():
    with Session(engine) as session:
        yield session

# for single use sessions
def get_session_direct() -> Session:
    return Session(engine)