import logging
import os
from typing import List, Tuple

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config

FILE_SUFFIX = [".pdf",".doc",".docx",".txt"]

def get_file_list(folder_path:str):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1] in FILE_SUFFIX:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:

    text = ""
    page_numbers = []

    for page_number,page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split('\n')))
        else:
            logging.warning(f"No text found on page {page_number}")
    return text, page_numbers

def process_text_with_splitter(text:str) -> list[str]:

    # create text splitter
    text_spliter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " ",".",""],
        chunk_size = Config.CHUNK_SIZE,
        chunk_overlap = Config.CHUNK_OVERLAP,
        length_function = len,
    )

    chunks = text_spliter.split_text(text)
    logging.debug(f"Text split into {len(chunks)} chunks")
    return chunks

def data_loader(file_path):
    pdf_file_list = get_file_list(file_path)
    all_chunks = []
    for pdf in pdf_file_list:
        pdf = PdfReader(pdf)
        text, page_numbers = extract_text_with_page_numbers(pdf)
        chunks = process_text_with_splitter(text)
        all_chunks.extend(chunks)
    return all_chunks

# unit test
# if __name__ == "__main__":
#     loader_path = "./assets"
#     chunks_test = data_loader(loader_path)
#     print(chunks_test)