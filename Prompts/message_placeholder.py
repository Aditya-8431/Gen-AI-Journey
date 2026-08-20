from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

#chat Template
chat_template=ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

#load chat history
chat_history=[]
with open('chat_history.txt','r') as f:
    chat_history.extend(f.readlines())

print("Chat History: ", chat_history)
# Create Prompt
promt=chat_template.invoke({ 'chat_history': chat_history, 'query': 'What is the status of my order?'})
print("Prompt: ", promt)