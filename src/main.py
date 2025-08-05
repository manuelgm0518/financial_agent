from agents import data_cleaner_agent
from agents import simple_search_agent

from agno.playground import Playground


playground_app = Playground(agents=[data_cleaner_agent, simple_search_agent])

app = playground_app.get_app()

if __name__ == "__main__":
    playground_app.serve("main:app", reload=True)