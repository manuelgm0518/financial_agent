from agents import data_cleaner_agent
from agents import simple_search_agent

from agno.playground import Playground



# if __name__ == "__main__":
#     data_cleaner_agent.print_response()



playground_app = Playground(agents=[simple_search_agent])

app = playground_app.get_app()

if __name__ == "__main__":
    playground_app.serve("main:app", reload=True)