from google import genai
import io
import os
import sys

API_KEY = "[YOUR KEY]"

# read your key from a text file. don't commit this! the repo is public.
with open("api_key.txt", "r") as file:
    API_KEY = file.read()

PROMPT = """
You are a robot meant to help with household tasks.
Given the following list of locations and a task, provide the location(s) from this list that the robot should move to in order to complete the task:
- KITCHEN
- HALLWAY
- LIVING ROOM
The robot is currently not in any of the locations on the list. The task is this:
"Put the dishes in the sink."
"""

# module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests'))
# sys.path.insert(0, module_dir)

# Initialize Gemini API
client = genai.Client(api_key=API_KEY)

print(PROMPT)
print("=====================")

print("Sending to Gemini... this may take a moment.")
try:
    # Send the prompt and the image to the model
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=PROMPT
    )

    print("------------------------------------------------")

    # Try to print text for debugging
    try:
        print(f"   Response text: {response.text}")
    except Exception:
        pass # No text part

except Exception as e:
    print(f"An error occurred with the Gemini API: {e}")