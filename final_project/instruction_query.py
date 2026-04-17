from google import genai
import cv2

API_KEY = "[YOUR KEY]"

LOC_PROMPT = """
You are a robot meant to help with household tasks.
Given the following list of locations and a task, provide the location(s) from this list that the robot should move to in order to complete the task:
- KITCHEN
- HALLWAY
- LIVING ROOM
The robot is currently not in any of the locations on the list. The task is this:
"Put the dishes in the sink."
"""

GRIP_PROMPT = """
You are a Stretch 3 robot that is trying to grab an object. Given this image taken from a camera mounted on the gripper, determine whether or not the gripper has successfully grasped the object.
Give your answer as YES or NO.
"""

GRIP_IMG = cv2.imread('grip.png')

def gemini_init():
    # read your key from a text file. don't commit this! the repo is public.
    with open("api_key.txt", "r") as file:
        API_KEY = file.read()

    # Initialize Gemini API
    return genai.Client(api_key=API_KEY)

def prompt_gemini(client, mode):
    prompt = None
    if mode is "loc":
        prompt = LOC_PROMPT
    elif mode is "grip":
        prompt = [GRIP_PROMPT, GRIP_IMG]

    print(prompt)
    print("=====================")

    print("Sending to Gemini... this may take a moment.")
    try:
        # Send the prompt and the image to the model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        print("------------------------------------------------")

        # Try to print text for debugging
        try:
            print(f"   Response text: {response.text}")
        except Exception:
            pass # No text part

        return response

    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")