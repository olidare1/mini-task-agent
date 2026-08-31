import os
import shutil
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Define the tools the agent can use
@tool
def create_folder(folder_name: str) -> str:
    """Erstellt einen neuen Ordner auf dem System."""
    os.makedirs(folder_name, exist_ok=True)
    return f"Ordner '{folder_name}' wurde erfolgreich erstellt."

@tool
def create_file(file_name: str, content: str) -> str:
    """Erstellt eine Datei mit dem angegebenen Inhalt."""
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Datei '{file_name}' wurde erfolgreich mit Inhalt gefüllt."

@tool
def check_disk_space(path: str = ".") -> str:
    """Überprüft den freien Speicherplatz auf der Festplatte."""
    total, used, free = shutil.disk_usage(path)
    free_gb = round(free / (1024**3), 2)
    return f"Freier Speicherplatz: {free_gb} GB."

tools = [create_folder, create_file, check_disk_space]

# 2. Model & Prompt 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein hilfreicher Assistent, der Aufgaben auf dem Computer des Nutzers ausführen kann. Nutze die bereitgestellten Tools schrittweise."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 3. Create & execute agent 
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    task = "Erstelle einen Ordner 'Projekt_Daten', erstelle darin eine Datei 'info.txt' mit dem Text 'Agent gestartet' und prüfe danach den freien Speicherplatz."
    agent_executor.invoke({"input": task})