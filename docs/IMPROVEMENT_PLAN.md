Currently, the project only allows for the following simple sequence: User input -> plan out the tasks -> keep executing -> assess completion -> done.

However, we need to add the following functionalities to the planner style agent:
    1. User input can be given at any time, it should be asynchronous to the ongoing task (you need to figure out how to do this (eg: via ui and pub-sub?), since langgraphs only take user input from start nodes.)
    2. The agent maintains a clear goal (which is decided based on user input and can be shifted using user input but not the ai task execution), is able to assess that whatever upcoming/modified plans it is generating stick to that goal, and the ongoing task execution is satisfying both the plan and the goal.
    3. aside from MCP, the langgraph is also able to use langgraph.tools (i.e. custom made tools in the code, eg: shell command firing)

NEW GOALS:
    1. the agent must have the tools to fire shell commands locally (you don't need any other tools really, because everything can be done using that.)
    2. You must ensure that you satisfy the REQUIREMENTS BELOW and confirm that the agent is able to generate and deploy a detailed website using the given mcp and shell tool.

REQUIREMENTS:
    1. You must not write any tests or execute python code in a string for the implementation. The functionality must only be tested by running the langgraph (langgraph dev) and either sending api requests or running the async functions separately.
    2. You must update documentation upon every change and keep no more than 1 .md file for the enhanced planner
    3. You must ensure you test every aspect of your changes using the methods mentioned above and see to it that the agent delivers real results.
