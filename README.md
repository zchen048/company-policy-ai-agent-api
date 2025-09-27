# RAG Company Policy Query Agent API (Work in progress)

<br />

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-description">Project Description</a>
    </li>
    <li>
      <a href="#built-with">Built With</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
  </ol>
</details>

<br />

## Project Description
<p>Company policies are often scattered across multiple documents, making it hard for employees to quickly find accurate answers. This project solves that problem by providing an AI-powered agent that retrieves, understands, and summarizes policy information in response to natural language queries. Policy data used in this project was generated using ChatGPT.</p>
 
<p>Built as an API, it allows seamless integration into internal tools or chat interfaces, enabling users to get fast, grounded, and reliable answers without manually searching through documents.</p>

<br />

## Built With
This section list any major frameworks/technologies used to build this project.

- [LangGraph](https://www.langchain.com/langgraph)
- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLModel](https://sqlmodel.tiangolo.com/)
- [Chroma](https://www.trychroma.com/)

<br />

## Getting Started
1. Clone the repo
2. Create a virtual environment
3. Install project dependencies
   ```sh
   pip install -r requirements.txt
   ```
4. Get a [Groq API Key](https://groq.com/) and save in a `.env` file
5. Run the development server:
   ```sh
   fastapi dev main.py --port PORT_NUMBER
   ```

<br />

## Usage
Once the API is running, you can test the RAG Query Agent in two ways:
1. Using a tool like Postman  
   - Send a POST request to the `/message/query` endpoint with your input JSON to query the agent
   - Example payload:
     ```json
     {
       "user input": "What is the leave policy?"
     }
     ```
2. Using the interactive API docs
   - Navigate to http://localhost:PORT_NUMBER/docs in your browser.  
   - This provides a Swagger UI where you can try out all available endpoints directly.
