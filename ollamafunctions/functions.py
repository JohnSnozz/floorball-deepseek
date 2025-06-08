import ollama
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fbfunctions import currentteam, teambyyear, lineupsofgame, teamsofclub, clubidofclubname
import json
import re

def orchestration(text, debug=False):
    """Orchestrate function calls based on user input using deepseek-r1:14b"""
    
    prompt = f"""
You are a function dispatcher for a football/soccer data system. Based on the user's question, determine which function to call and extract the parameters.

Available functions:
1. currentteam(team_name) - Get current team information
2. teambyyear(team_name, year) - Get team information by year  
3. lineupsofgame(gameid, team=None) - Get lineups of a game, optionally filtered by team
4. teamsofclub(teamname) - Get all teams of a club
5. clubidofclubname(teamname) - Get club ID from club name

User question: "{text}"

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
        else:
            return "Could not determine appropriate function to call"
            
    except Exception as e:
        return f"Error in orchestration: {str(e)}"