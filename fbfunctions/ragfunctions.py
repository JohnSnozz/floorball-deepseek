import ollama
import os
import glob
import PyPDF2

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

def rulesquestion(question):
    """Get information about the ruleset using RAG on context PDFs"""
    
    # Load context from PDFs
    context = load_pdf_context_local()
    
    # Rest bleibt gleich...