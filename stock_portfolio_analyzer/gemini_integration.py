import os
import google.generativeai as genai
from google.api_core.exceptions import InvalidArgument
import streamlit as st

def setup_gemini():
    """
    Set up the Gemini API client
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Hardcoded API key as requested
        api_key = "AIzaSyC0rB8umNykwwRVIv8GOo06VP-v10h1qeA"
        
        # Configure the Gemini API client
        genai.configure(api_key=api_key)
        
        return True
    except Exception as e:
        print(f"Error setting up Gemini: {e}")
        return False

def get_gemini_recommendation(prompt):
    """
    Get investment recommendations from Gemini
    
    Args:
        prompt (str): The prompt to send to Gemini
        
    Returns:
        str: The recommendation from Gemini or an error message
    """
    # Set up Gemini with hardcoded API key
    api_setup_success = setup_gemini()
    
    try:
        # Configure the generation parameters
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        # Initialize the Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Return the response text
        return response.text
    except InvalidArgument as e:
        return f"There was an issue with the Gemini API: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
