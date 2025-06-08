import warnings
import logging

# Alle PDF-related warnings unterdrücken
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*CropBox.*")

# PyPDF2 logging unterdrücken
logging.getLogger("PyPDF2").setLevel(logging.CRITICAL)
logging.getLogger("pdfplumber").setLevel(logging.CRITICAL)



import ollama
import os
import glob
import PyPDF2
import pdfplumber

import re


def load_pdf_context_local(context_folder="./context"):
    """Load all PDF files from context folder and return combined text"""
    
    context_text = ""
    pdf_files = glob.glob(os.path.join(context_folder, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {context_folder}")
        return ""
    
    for pdf_file in pdf_files:
        try:
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pdf_text = ""
                
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() + "\n"
                
                context_text += f"\n--- Content from {os.path.basename(pdf_file)} ---\n"
                context_text += pdf_text
                print(f"Loaded: {os.path.basename(pdf_file)}")
                
        except Exception as e:
            print(f"Error loading {pdf_file}: {str(e)}")
    
    return context_text

def load_pdf_context_chunked(context_folder="./context"):
    """Load PDFs and split into article chunks"""
    
    context_chunks = []
    pdf_files = glob.glob(os.path.join(context_folder, "*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text() or ""
                
                # Split by articles (Art. X.X pattern)
                articles = re.split(r'(?=Art\.\s+\d+\.\d+)', full_text)
                
                for i, article in enumerate(articles):
                    if article.strip() and len(article) > 100:  # Filter kurze chunks
                        context_chunks.append({
                            'source': os.path.basename(pdf_file),
                            'article_num': i,
                            'content': article.strip()[:2000]  # Limit chunk size
                        })
                        
                print(f"Extracted {len([a for a in articles if len(a) > 100])} articles from {os.path.basename(pdf_file)}")
                        
        except Exception as e:
            print(f"Error: {e}")
    
    return context_chunks


def rulesquestion(question):
    """Answer using chunked context"""
    
    chunks = load_pdf_context_chunked()
    
    # Combine relevant chunks (oder später: semantic search)
    context = "\n\n".join([f"[{chunk['source']}]\n{chunk['content']}" 
                          for chunk in chunks[:10]])  # Erste 10 chunks
    
    prompt = f"""
    Based on the floorball rules below, answer the question.

    Rules context:
    {context}

    Question: {question}
    """

    try:
        response = ollama.chat(
            model='deepseek-r1:14b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        answer = response['message']['content'].strip()
        
        # Stop ollama and clear cache
        try:
            ollama.stop('deepseek-r1:14b')
            os.system('ollama ps | grep -v NAME | awk \'{print $1}\' | xargs -r ollama stop')
        except:
            pass
            
        return answer
        
    except Exception as e:
        return f"Error answering rules question: {str(e)}"