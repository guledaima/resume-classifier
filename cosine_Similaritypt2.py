from sentence_transformers import SentenceTransformer
from transformers import XLNetTokenizer, XLNetModel
from transformers import BertTokenizer,BertModel
import gradio as gr
import numpy as np
import io
import pdfplumber
import langsmith
import torch
from langsmith import traceable


LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="Resume Classifier"

def read_pdf(file):
    text = []

    try:
         with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text())
    except Exception as e:
          print(f"An error occurred while processing the file: {e}")
          return None

    return " ".join(text)  # Combine all text into a single string


#intializing the sentence transformer/sbert model
s_model= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#intializing bert model
b_Tokenizier=BertTokenizer.from_pretrained('bert-base-uncased')
b_model=BertModel.from_pretrained('bert-base-uncased')

#intializing xl model
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased')

#generate embeddings

def sbert_embeddings(text):
    embeddings=s_model.encode(text)
    return embeddings
    


def bert_embeddings(text, chunk_size=512):
    tokens = b_Tokenizier.tokenize(text)
    embeddings = []

    # Process the text in chunks
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_inputs = b_Tokenizier.convert_tokens_to_ids(chunk_tokens)
        chunk_inputs = torch.tensor([chunk_inputs])
        
        attention_mask = torch.ones_like(chunk_inputs)
        
        with torch.no_grad():
            chunk_outputs = b_model(input_ids=chunk_inputs, attention_mask=attention_mask)
            chunk_embedding = chunk_outputs.last_hidden_state.mean(dim=1)  # Average pooling
            embeddings.append(chunk_embedding)
    
    # Average all chunk embeddings to get the final embedding
    final_embedding = torch.stack(embeddings).mean(dim=0).squeeze().numpy()
    return final_embedding


def xlnet_embeddings(text, max_length=512):
    tokens = xlnet_tokenizer.encode(text, add_special_tokens=True)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    embeddings = []

    for chunk in chunks:
        inputs = torch.tensor(chunk).unsqueeze(0)  
        with torch.no_grad():
            outputs = xlnet_model(inputs)[0]  # Get last hidden state
        chunk_embedding = outputs.mean(dim=1).squeeze().numpy()  # Average embeddings
        embeddings.append(chunk_embedding)

    # Combine chunk embeddings (e.g., by averaging)
    combined_embedding = np.mean(embeddings, axis=0)
    return combined_embedding

    
#calculate cosine similarity
@traceable(run_type="similarity", name="Cosine Similarity")
def calculate_similarity(embedding1, embedding2):
    embedding1=embedding1.flatten()
    embedding2=embedding2.flatten()
    similarity= np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return float(similarity)

def hiring_decision(similarity_score, threshold=0.5):
    if (similarity_score>=threshold):
        decision="hired"
    else:
        decision="not hired"
    return decision


@traceable
def predict_hiring(resume_file, job_description):
    results=[]
    decisions=[]
    for resume_file in resume_file:
            resume_text=read_pdf(resume_file)
            #bert
            bert_resume_embedding=bert_embeddings(resume_text)
            bert_job_description_embedding=bert_embeddings(job_description)
            bert_similarity=calculate_similarity(bert_resume_embedding,bert_job_description_embedding)
            #sbert
            sbert_resume_embedding=sbert_embeddings(resume_text)
            sbert_job_description_embedding=sbert_embeddings(job_description)
            sbert_similarity=calculate_similarity(sbert_resume_embedding,sbert_job_description_embedding)
            #t5
            xl_resume_embedding=xlnet_embeddings(resume_text)
            xl_job_description_embedding=xlnet_embeddings(job_description)
            xl_similarity=calculate_similarity(xl_resume_embedding,xl_job_description_embedding)

            results.append(f"{resume_file.name}:\n" 
                           f"bert similarity score: {bert_similarity:.4f}\n"
                           f"sbert similartity score: {sbert_similarity:.4f}\n"
                           f"xlnet similartity score: {xl_similarity:.4f}")

            bert_decision=hiring_decision(bert_similarity)
            sbert_decision=hiring_decision(sbert_similarity)
            xlnet_decision=hiring_decision(xl_similarity)

            decisions.append(f"{resume_file.name}:\n" 
                           f"bert decision:  {bert_decision}\n"
                           f"sbert decision: {sbert_decision}\n"
                           f"xlnet decision: {xlnet_decision}")
    
    return "\n".join(results) , decisions

with gr.Blocks() as interface:
    inputs=[gr.Files(label="Resume file"),gr.Text(label="job description")]
    outputs=[gr.Text(label="similarity score"),gr.Text(label="hiring prediction")]
    title="Resume classification"
    description="Enter a resume and a job description to see if the candidate is likely to be hired."
    show=gr.Button(value="hiring probability")
    show.click(predict_hiring, inputs, outputs)


interface.launch()
