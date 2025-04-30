import os
import pandas as pd
import fitz
import re
from spacy.lang.en import English

PDF_PATH = "coral-doco.pdf"

if not os.path.exists(PDF_PATH):
    print("The file doesn't exist. No PDF has been loaded.")
else:
    print(f"File {PDF_PATH} exists.")

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()

    return cleaned_text

# Open PDf and get lines/pages
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
    list[dict]: A list of dictionaries, each containing the page number
    (adjusted), character count, word count, sentence count, token count, and the extracted text for each page.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in enumerate(doc):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "text": text
                               })
    return pages_and_texts

pages_and_texts = open_and_read_pdf(PDF_PATH)

df = pd.DataFrame(pages_and_texts)

nlp = English()

nlp.add_pipe("sentencizer")

for item in pages_and_texts:
    item["sentences"] = list(nlp(item["text"]).sents)

    # Make sure all sentences are strings
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]

    # count the sentences
    item["page_sentence_count_spacy"] = len(item["sentences"])

num_sentence_chunk_size = 10

def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of slize slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into the two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

for item in pages_and_texts:
    item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                         slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])

# Split each chunk into its own item
pages_and_chunks = []
for item in pages_and_texts:
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]
        
        # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
        
        pages_and_chunks.append(chunk_dict)

min_token_length = 30
pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device="cpu") 

embedding_model.to("cpu")

# Create embeddings one by one on the GPU
for item in pages_and_chunks_over_min_token_len:
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])

text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
