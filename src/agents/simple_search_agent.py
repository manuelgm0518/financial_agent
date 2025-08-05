from agno.agent import Agent
from agno.tools.csv_toolkit import CsvTools
import os

# Dynamically load all CSV files from the data_cleaned directory
data_cleaned_dir = "data_cleaned"
csv_files = [
    os.path.join(data_cleaned_dir, f)
    for f in os.listdir(data_cleaned_dir)
    if f.endswith(".csv")
]

simple_search_agent = Agent(
    name="Simple Search Agent",
    description="Search for data in the data_cleaned CSV files",
    tools=[CsvTools(csvs=csv_files)],
    instructions=[
        "Load data from the data_cleaned CSV files.",
        "Check the columns in each file.",
        "Choose the right file based on the user's question.",
        "Explain your reasoning for the user.",
        "Add the calculated value to the response.",
        "Answer briefly and to the point.",
        "Highlight the final answer.",
        "Available CSV files: " + ", ".join([os.path.basename(f) for f in csv_files]) if csv_files else "No CSV files found"
    ],
    show_tool_calls=True,
    markdown=True,
)