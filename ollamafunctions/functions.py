import ollama
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fbfunctions import currentteam, teambyyear, lineupsofgame, teamsofclub, clubidofclubname, rulesquestion
import json
import re
import os
import glob
import PyPDF2
from pathlib import Path

def load_pdf_context(context_folder="./context"):
    """Load all PDF files from context folder and return combined text (cached)"""
    
    global _pdf_cache
    if _pdf_cache is not None:
        return _pdf_cache
    
    import warnings
    warnings.filterwarnings("ignore")
    
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
    
    # Cache the result
    _pdf_cache = context_text
    return context_text

def translate_to_english(text, source_lang="German"):
    """Translate text from source language to English using gemma3-translator"""
    
    # Load context from PDFs
    context = load_pdf_context()
    
    prompt = f"""Translate the following {source_lang} text to English.

Context: {context}

Text to translate: "{text}"

Provide only the English translation, no explanations."""
    
    try:
        response = ollama.chat(
            model='zongwei/gemma3-translator:4b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        translated_text = response['message']['content'].strip()
        
        # Stop ollama and clear cache
        try:
            ollama.stop('zongwei/gemma3-translator:4b')
            os.system('ollama ps | grep -v NAME | awk \'{print $1}\' | xargs -r ollama stop')
        except:
            pass
            
        return translated_text
        
    except Exception as e:
        return f"Translation error: {str(e)}"

def orchestration(text, debug=False, translate=True, source_lang="German"):
   """Orchestrate function calls based on user input using deepseek-r1:14b"""
   
   # Translate input if requested
   if translate:
       english_text = translate_to_english(text, source_lang=source_lang)
       if debug:
           print(f"Original ({source_lang}): {text}")
           print(f"Translated: {english_text}")
   else:
       english_text = text
       if debug:
           print(f"No translation - using original: {text}")
   
   prompt = f"""
You are a function dispatcher for a floorball data system. Based on the user's question, determine which function to call and extract the parameters.

Available functions:
1. currentteam(team_name) - Get current team information
2. teambyyear(team_name, year) - Get team information by year  
3. lineupsofgame(gameid, team=None) - Get lineups of a game, optionally filtered by team
4. teamsofclub(teamname) - Get all teams of a club
5. clubidofclubname(teamname) - Get club ID from club name
6. rulesquestion(question) - Get answers to questions about the current rules

User question: "{english_text}"

Respond with EXACTLY this JSON format:
{{
   "function": "function_name",
   "parameters": {{
       "param1": "value1",
       "param2": "value2"
   }}
}}

If you can't determine a function, respond with:
{{
   "function": "none",
   "parameters": {{}}
}}
"""

   try:
       response = ollama.chat(
           model='deepseek-r1:14b',
           messages=[{'role': 'user', 'content': prompt}]
       )
       
       response_text = response['message']['content'].strip()
       
       if debug:
           print(f"Raw response: {response_text}")
       
       # Extract JSON from the response (skip the <think> part)
       json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
       if json_match:
           json_text = json_match.group(0)
           decision = json.loads(json_text)
       else:
           return f"No JSON found in response"
       
       # Call the appropriate function
       function_name = decision.get('function')
       params = decision.get('parameters', {})
       
       if function_name == 'currentteam':
           return currentteam(params.get('team_name'))
       elif function_name == 'teambyyear':
           return teambyyear(params.get('team_name'), params.get('year'))
       elif function_name == 'lineupsofgame':
           return lineupsofgame(params.get('gameid'), params.get('team'))
       elif function_name == 'teamsofclub':
           return teamsofclub(params.get('teamname'))
       elif function_name == 'clubidofclubname':
           return clubidofclubname(params.get('teamname'))
       elif function_name == 'rulesquestion':
           return rulesquestion(params.get('question'))
       else:
           return "Could not determine appropriate function to call"
           
   except Exception as e:
       return f"Error in orchestration: {str(e)}"