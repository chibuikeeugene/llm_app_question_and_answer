# LLM Q&A with any website  

A large language model that takes any website as an input together with a user query prompt and returns a suitable human response. This program uses a **llama3 model locally hosted** to return human readable and understandable results.  

## Dependencies and packages  

1. python = "^3.10"
2. streamlit = "^1.36.0"
3. langchain = "^0.2.6"
4. llama 3
5. huggingface-hub = "^0.23.4"
6. python-dotenv = "^1.0.1"
7. langchain-community = "^0.2.6"
8. langchain-ollama = "^0.1.0"
9. beautifulsoup4 = "^4.12.3"
10. faiss-cpu = "^1.8.0.post1"
11. sphinx = "^8.0.2"
12. loguru = "^0.7.2"
13. fire = "^0.6.0"

## To run the application

1. install llama3 and run server locally - using the command -  ollama run llama3
2. Install the package dependencies using poetry install command. Ensure you do this in a virual environment.
3. navigate to the q_and_a sub-folder and run the command -  streamlit run main.py
4. Good luck !

## Snapshots

### Q&A architecture

![Q&A app schema](<LangSmith Question & Answer.drawio.png>)

### Live view of our application in workmode with Streamlit as the frontend

![Streamlit app 1](<Screenshot 2024-08-23 at 11.49.52 pm.png>)

![Streamlit app 2](<Screenshot 2024-08-23 at 11.49.59 pm.png>)
