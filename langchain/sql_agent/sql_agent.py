from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# https://docs.langchain.com/oss/python/langchain/sql-agent

# Initialize an LLM
model = init_chat_model('deepseek-chat')

# Get the database, store it locally
# https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip
db = SQLDatabase.from_uri('sqlite:///chinook.db')

# Create the tools
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

for tool in tools:
    print(f'{tool.name}: {tool.description}')

# Create the agent
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)

# Run the agent
question = 'Which genre on average has the longest tracks?'

for step in agent.stream(
    {'messages': [{'role': 'user', 'content': question}]},
    stream_mode='values',
):
    step['messages'][-1].pretty_print()
