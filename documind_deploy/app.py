# Import the gradio library for creating web-based user interfaces.
import gradio as gr
# Import the os module to interact with the operating system, like accessing environment variables.
import os
# Import SentenceTransformer for generating text embeddings.
from sentence_transformers import SentenceTransformer
# Import faiss for efficient similarity search on vector embeddings.
import faiss
# Import numpy for numerical operations, especially with arrays.
import numpy as np
# Import the OpenAI client to interact with LLMs, including NVIDIA's models.
from openai import OpenAI
# Import PyPDF2 for reading and extracting text from PDF files.
import PyPDF2
# Import Document from docx to work with Word (.docx) files.
from docx import Document

# Document Loader class to handle loading content from various document types.
class DocumentLoader:
    # Method to load a document based on its file path and type.
    def load(self, file_path, file_type):
        # Check if the file type is 'pdf'.
        if file_type == 'pdf':
            # If it's a PDF, call the internal PDF loading method.
            return self._load_pdf(file_path)
        # Check if the file type is 'docx'.
        elif file_type == 'docx':
            # If it's a DOCX, call the internal DOCX loading method.
            return self._load_docx(file_path)
        # Check if the file type is 'txt'.
        elif file_type == 'txt':
            # If it's a TXT, call the internal TXT loading method.
            return self._load_txt(file_path)
        # If the file type is not supported.
        else:
            # Raise an error for unsupported file types.
            raise ValueError(f"Unsupported file type: {file_type}")

    # Internal method to load text from a PDF file.
    def _load_pdf(self, file_path):
        # Initialize an empty string to store extracted text.
        text = ""
        # Open the PDF file in binary read mode.
        with open(file_path, 'rb') as file:
            # Create a PdfReader object to read the PDF.
            pdf_reader = PyPDF2.PdfReader(file)
            # Iterate through each page in the PDF.
            for page in pdf_reader.pages:
                # Extract text from the page and append it to the text string, followed by a newline.
                text += page.extract_text() + "\n"
        # Return the accumulated text from the PDF.
        return text

    # Internal method to load text from a DOCX file.
    def _load_docx(self, file_path):
        # Open the DOCX document.
        doc = Document(file_path)
        # Initialize an empty string to store extracted text.
        text = ""
        # Iterate through each paragraph in the document.
        for paragraph in doc.paragraphs:
            # Append the text of each paragraph, followed by a newline.
            text += paragraph.text + "\n"
        # Return the accumulated text from the DOCX file.
        return text

    # Internal method to load text from a plain TXT file.
    def _load_txt(self, file_path):
        # Open the text file in read mode with UTF-8 encoding.
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the entire content of the file.
            text = file.read()
        # Return the read text.
        return text

# Text Chunker class to divide long texts into smaller, overlapping segments.
class TextChunker:
    # Constructor to initialize chunk size and overlap.
    def __init__(self, chunk_size=500, overlap=50):
        # Set the maximum size for each text chunk.
        self.chunk_size = chunk_size
        # Set the amount of characters that overlap between consecutive chunks.
        self.overlap = overlap

    # Method to chunk a given text.
    def chunk(self, text):
        # Initialize an empty list to store the chunks.
        chunks = []
        # Set the starting position for chunking.
        start = 0
        # Loop through the text until the start position exceeds its length.
        while start < len(text):
            # Calculate the end position for the current chunk.
            end = start + self.chunk_size
            # Extract the chunk of text.
            chunk = text[start:end]
            # If the chunk is not empty after stripping whitespace.
            if chunk.strip():
                # Add the cleaned chunk to the list.
                chunks.append(chunk.strip())
            # Move the start position forward by chunk_size minus overlap to create the next chunk.
            start += (self.chunk_size - self.overlap)
        # Return the list of generated chunks.
        return chunks

# Embedder class to convert text into numerical vector embeddings.
class Embedder:
    # Constructor to initialize the SentenceTransformer model.
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Print a message indicating that the embedding model is loading.
        print("Loading embedding model...")
        # Load the specified SentenceTransformer model.
        self.model = SentenceTransformer(model_name)
        # Print a message confirming that the embedding model has loaded.
        print("✅ Embedding model loaded!")

    # Method to generate embeddings for a list of texts.
    def embed(self, texts):
        # Encode the texts into embeddings, suppressing the progress bar.
        embeddings = self.model.encode(texts, show_progress_bar=False)
        # Return the generated embeddings.
        return embeddings

# VectorStore class to store and search for document embeddings using FAISS.
class VectorStore:
    # Constructor to initialize the vector store with a given dimension.
    def __init__(self, dimension=384):
        # Store the dimension of the embeddings.
        self.dimension = dimension
        # Initialize a FAISS index for flat L2 distance search.
        self.index = faiss.IndexFlatL2(dimension)
        # Initialize an empty list to store the original text chunks.
        self.chunks = []

    # Method to add new embeddings and their corresponding chunks to the store.
    def add(self, embeddings, chunks):
        # Convert embeddings to a NumPy array of float32 type, as required by FAISS.
        embeddings = np.array(embeddings).astype('float32')
        # Add the embeddings to the FAISS index.
        self.index.add(embeddings)
        # Extend the list of stored chunks with the new chunks.
        self.chunks.extend(chunks)
        # Print a confirmation message showing how many chunks were added.
        print(f"✅ Added {len(chunks)} chunks to vector store!")

    # Method to search the vector store for chunks similar to a query embedding.
    def search(self, query_embedding, top_k=3):
        # Convert the query embedding to a NumPy array of float32, suitable for FAISS.
        query_embedding = np.array([query_embedding]).astype('float32')
        # Perform a search in the FAISS index for the top_k nearest neighbors.
        distances, indices = self.index.search(query_embedding, top_k)
        # Initialize an empty list to store the search results.
        results = []
        # Iterate through the indices of the found neighbors.
        for idx in indices[0]:
            # Ensure the index is valid within the range of stored chunks.
            if idx < len(self.chunks):
                # Add the corresponding chunk to the results.
                results.append(self.chunks[idx])
        # Return the relevant chunks.
        return results

    # Method to clear all data from the vector store.
    def clear(self):
        # Re-initialize the FAISS index, effectively clearing all stored vectors.
        self.index = faiss.IndexFlatL2(self.dimension)
        # Clear the list of stored text chunks.
        self.chunks = []
        # Print a confirmation message that the vector store has been cleared.
        print("✅ Vector store cleared!")

# NVIDIA LLM class to interact with a Large Language Model hosted by NVIDIA.
class NvidiaLLM:
    # Constructor to initialize the LLM client.
    def __init__(self):
        # Retrieve the NVIDIA API key from environment variables.
        api_key = os.environ.get('NVIDIA_API_KEY')
        # Check if the API key is not found.
        if not api_key:
            # Raise a ValueError if the API key is missing.
            raise ValueError("⚠️ NVIDIA_API_KEY not found! Add it in Space Settings → Repository secrets")

        # Initialize the OpenAI client with NVIDIA's API base URL and the retrieved API key.
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        # Print a message confirming successful connection to the NVIDIA LLM.
        print("✅ NVIDIA LLM connected!")

    # Method to generate text based on a given prompt.
    def generate(self, prompt, max_tokens=1000):
        # Start a try-except block to handle potential errors during API calls.
        try:
            # Make a chat completion request to the LLM.
            response = self.client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct", # Specify the LLM model to use.
                messages=[{"role": "user", "content": prompt}], # Provide the user prompt.
                max_tokens=max_tokens, # Set the maximum number of tokens for the generated response.
                temperature=0.2 # Set the creativity/randomness of the response (lower is more deterministic).
            )
            # Return the content of the first message choice from the LLM's response.
            return response.choices[0].message.content
        # Catch any exceptions that occur during the API call.
        except Exception as e:
            # Return an error message including the exception details.
            return f"❌ Error: {str(e)}"

# RAG Pipeline class, integrating all components for Retrieval Augmented Generation.
class RAGPipeline:
    # Constructor to initialize all RAG components.
    def __init__(self):
        # Initialize the DocumentLoader.
        self.loader = DocumentLoader()
        # Initialize the TextChunker with specified chunk size and overlap.
        self.chunker = TextChunker(chunk_size=500, overlap=50)
        # Initialize the Embedder.
        self.embedder = Embedder()
        # Initialize the VectorStore.
        self.vector_store = VectorStore()
        # Initialize the NvidiaLLM.
        self.llm = NvidiaLLM()
        # Initialize a variable to hold the current document's text.
        self.current_document = None
        # Print a message confirming the RAG Pipeline is ready.
        print("✅ RAG Pipeline ready!")

    # Method to process an uploaded document.
    def process_document(self, file_path, file_type):
        # Load the text content from the document using the loader.
        text = self.loader.load(file_path, file_type)
        # Store the loaded document text.
        self.current_document = text
        # Chunk the loaded text.
        chunks = self.chunker.chunk(text)
        # Generate embeddings for the chunks.
        embeddings = self.embedder.embed(chunks)
        # Clear any existing data in the vector store.
        self.vector_store.clear()
        # Add the new embeddings and chunks to the vector store.
        self.vector_store.add(embeddings, chunks)
        # Return a success message indicating the number of chunks created.
        return f"✅ Document processed! Created {len(chunks)} chunks."

    # Method to answer a question based on the processed document.
    def answer_question(self, question):
        # Check if a document has been uploaded.
        if not self.current_document:
            # If no document, return an error message and empty list.
            return "❌ Please upload a document first!", []

        # Generate an embedding for the user's question.
        question_embedding = self.embedder.embed([question])[0]
        # Search the vector store for the most relevant chunks.
        relevant_chunks = self.vector_store.search(question_embedding, top_k=3)

        # Check if no relevant chunks were found.
        if not relevant_chunks:
            # Return an error message if no relevant information is found.
            return "❌ No relevant information found.", []

        # Format the relevant chunks into a context string for the LLM.
        context = "\n\n".join([f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(relevant_chunks)])
        # Construct the prompt for the LLM, instructing it to answer based only on provided chunks.
        prompt = f"""You are a document analysis assistant. Answer based ONLY on provided chunks.

Document Chunks:
{context}

Question: {question}

Answer:"""

        # Generate an answer using the LLM based on the prompt.
        answer = self.llm.generate(prompt)
        # Return the generated answer and the relevant chunks used.
        return answer, relevant_chunks

    # Method to summarize the current document.
    def summarize(self):
        # Check if a document has been uploaded.
        if not self.current_document:
            # If no document, return an error message.
            return "❌ Please upload a document first!"

        # Take a sample of the document text (first 3000 characters) for summarization.
        text_sample = self.current_document[:3000]
        # Construct the prompt for the LLM to summarize the text.
        prompt = f"""Summarize in 3-5 sentences:

{text_sample}

Summary:"""

        # Generate the summary using the LLM with a max token limit.
        return self.llm.generate(prompt, max_tokens=300)

    # Method to extract topics from the current document.
    def extract_topics(self):
        # Check if a document has been uploaded.
        if not self.current_document:
            # If no document, return an error message.
            return "❌ Please upload a document first!"

        # Take a sample of the document text (first 3000 characters) for topic extraction.
        text_sample = self.current_document[:3000]
        # Construct the prompt for the LLM to extract topics.
        prompt = f"""Extract 5-7 main topics as bullet points:

{text_sample}

Topics:"""

        # Generate the topics using the LLM with a max token limit.
        return self.llm.generate(prompt, max_tokens=300)

# Print an initialization message for DocuMind.
print("🚀 Initializing DocuMind...")
# Create an instance of the RAGPipeline.
pipeline = RAGPipeline()

# Gradio function to handle document uploads.
def upload_document(file):
    # Check if no file was selected.
    if file is None:
        # Return an error message if no file.
        return "❌ Please select a file!"

    # Get the file extension and convert it to lowercase.
    file_ext = file.name.split('.')[-1].lower()
    # Check if the file extension is not supported.
    if file_ext not in ['pdf', 'docx', 'txt']:
        # Return an error message for unsupported file types.
        return f"❌ Unsupported file type: {file_ext}"

    # Process the document using the pipeline and return the status message.
    return pipeline.process_document(file.name, file_ext)

# Gradio function to handle answering questions.
def ask_question(question):
    # Check if the question input is empty or just whitespace.
    if not question.strip():
        # Return an error if no question is entered.
        return "❌ Please enter a question!", ""

    # Get the answer and relevant chunks from the pipeline.
    answer, chunks = pipeline.answer_question(question)
    # Format the relevant chunks into a source string for display.
    sources = "\n\n".join([f"📄 Chunk {i+1}:\n{chunk[:300]}..." for i, chunk in enumerate(chunks)])
    # Return the answer and sources.
    return answer, sources

# Gradio function to trigger document summarization.
def summarize_doc():
    # Call the pipeline's summarize method.
    return pipeline.summarize()

# Gradio function to trigger topic extraction.
def extract_topics_doc():
    # Call the pipeline's extract_topics method.
    return pipeline.extract_topics()

# Define the Gradio web interface using gr.Blocks for a custom layout.
with gr.Blocks(title="DocuMind") as app:
    # Display a main title using Markdown.
    gr.Markdown("# 🧠 DocuMind: AI Document Assistant")
    # Display a subtitle/instruction using Markdown.
    gr.Markdown("Upload PDF, DOCX, or TXT files and ask questions!")

    # Create a tab for uploading documents.
    with gr.Tab("📤 Upload"):
        # Create a file input component for document upload.
        file_input = gr.File(label="Upload Document")
        # Create a button to trigger document processing.
        upload_btn = gr.Button("Process Document", variant="primary")
        # Create a textbox to display the upload status.
        upload_output = gr.Textbox(label="Status", lines=3)
        # Link the button click to the upload_document function, passing input and output components.
        upload_btn.click(upload_document, file_input, upload_output)

    # Create a tab for asking questions.
    with gr.Tab("💬 Questions"):
        # Create a textbox for the user to type their question.
        question_input = gr.Textbox(label="Your Question")
        # Create a button to get the answer.
        ask_btn = gr.Button("Get Answer", variant="primary")
        # Create a textbox to display the generated answer.
        answer_output = gr.Textbox(label="Answer", lines=5)
        # Create a textbox to display the sources (relevant chunks).
        sources_output = gr.Textbox(label="Sources", lines=8)
        # Link the button click to the ask_question function.
        ask_btn.click(ask_question, question_input, [answer_output, sources_output])

    # Create a tab for summarization.
    with gr.Tab("📝 Summary"):
        # Create a button to generate the summary.
        summarize_btn = gr.Button("Generate Summary", variant="primary")
        # Create a textbox to display the summary.
        summary_output = gr.Textbox(label="Summary", lines=6)
        # Link the button click to the summarize_doc function.
        summarize_btn.click(summarize_doc, None, summary_output)

    # Create a tab for topic extraction.
    with gr.Tab("🏷️ Topics"):
        # Create a button to extract topics.
        topics_btn = gr.Button("Extract Topics", variant="primary")
        # Create a textbox to display the extracted topics.
        topics_output = gr.Textbox(label="Topics", lines=6)
        # Link the button click to the extract_topics_doc function.
        topics_btn.click(extract_topics_doc, None, topics_output)

# This block ensures the Gradio app launches only when the script is executed directly.
if __name__ == "__main__":
    # Launch the Gradio application.
    app.launch()
