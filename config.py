import pandas as pd
import torch
import numpy as np
import pandas as pd
from sentence_transformers import util, SentenceTransformer

device = "cpu"

text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
embeddings.shape

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device=device) # choose the device to load the model to

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, 
                                   convert_to_tensor=True) 

    # Get dot product scores on embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)

    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict]=pages_and_chunks,
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    
    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print(pages_and_chunks[index]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")

query = "what is compass rfx?"

# Get just the scores and indices of top related results
scores, indices = retrieve_relevant_resources(query=query,
                                              embeddings=embeddings)
scores, indices

# Print out the texts of the top scores
print_top_results_and_scores(query=query,
                             embeddings=embeddings)

from ollama import chat
from IPython.display import Markdown, display

messages = [
    {'role': 'system', 'content': 'Your role is a helpful customer service chatbot for a software company called Coral Active.'},
    {'role': 'user', 'content': 'What is the black-scholes model?'}
]

stream = chat(
    model='llama3.2',
    messages=messages,
    stream=True,
)

# Initialize an empty string and create a display handle
output_text = ""
handle = display(Markdown(output_text), display_id=True)

# Stream tokens and update the same output cell
for chunk in stream:
    output_text += chunk['message']['content']
    handle.update(Markdown(output_text))

def prompt_formatter(query: str, 
                     context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """You are a helpful chatbot for Coral Active.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query   
    base_prompt = base_prompt.format(context=context, query=query)

    return base_prompt

messages = [
    {'role': 'system', 'content': 'Your role is a helpful customer service chatbot for a software company called Coral Active.'},
]

def ask_agent(query, messages, model='llama3.2'):
    # Get relevant resources
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings)
        
    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]
    
    formatted_query = prompt_formatter(query=query, context_items=context_items)

    messages.append({'role': 'user', 'content': formatted_query})
    
    stream = chat(
        model=model,
        messages=messages,
        stream=True,
    )

    output_text = ""
    handle = display(Markdown(output_text), display_id=True)
    
    # Stream tokens and update the same output cell
    for chunk in stream:
        output_text += chunk['message']['content']
        handle.update(Markdown(output_text))

def chat_agent(messages, model='llama3.2'):
    query = None
    while True:
        query = input()

        if query in ("", "/bye"):
            break

        ask_agent(query, messages, model)

chat_agent(messages, 'llama3.2')
