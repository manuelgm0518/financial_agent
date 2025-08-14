from agno.agent import Agent
from agno.storage.sqlite import SqliteStorage
from agno.tools.csv_toolkit import CsvTools
import os
from agno.models.openai import OpenAIChat

# Dynamically load all CSV files from the data_cleaned directory
data_cleaned_dir = "data_cleaned"
csv_files = [
    os.path.join(data_cleaned_dir, f)
    for f in os.listdir(data_cleaned_dir)
    if f.endswith(".csv")
]

storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/agent.db")


#Considerando la fecha de emisión y la fecha de vencimiento de las facturas, cuál es la factura con un tiempo de pago más largo?



simple_search_agent = Agent(
    model=OpenAIChat(id="o3-mini"),
    name="Simple Search Agent",
    description="Search for data in the data_cleaned CSV files",
    tools=[CsvTools(csvs=csv_files)],
    instructions=[
        "Load data from the data_cleaned CSV files.",
        "Check the columns in each file.",
        "Choose the right file or files based on the user's question.",
        "Explain your reasoning for the user.",
        "Add the calculated value to the response.",
        "Answer briefly and to the point.",
        "Highlight the final answer.",
        "Prefer tabular answers over text answers when possible.",
        "Available CSV files: " + ", ".join([os.path.basename(f) for f in csv_files]) if csv_files else "No CSV files found"
    ],
    show_tool_calls=True,
    markdown=True,
    storage=storage,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_runs=3,
)