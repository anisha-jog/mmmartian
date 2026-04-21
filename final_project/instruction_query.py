from google import genai
import cv2

API_KEY = "[YOUR KEY]"

LOC_PROMPT = """
You are a robot meant to help with household tasks.
Given the following list of locations and a task, provide the location(s) from this list that the robot should move to in order to complete the task:
- KITCHEN
- HALLWAY
- LIVING ROOM
Your response should only contain the list entry and no additional words (e.g. "KITCHEN").
The robot is currently not in any of the locations on the list. The task is this:
"""

SEC_LOC_PROMPT = """
You are a robot meant to help with household tasks.
You were given a previous task that you have partially completed. Now, you are in the correct location and holding an object to put in one of the following locations:
- SINK
- COUCH
Given your previous task and current location, provide the location from this list where you should move to place the object. Your response should only contain the list entry and no additional words (e.g. "SINK").
The task and current location are:
"""

GRIP_PROMPT = """
You are a Stretch 3 robot that is trying to grab an object. Given this image taken from a camera mounted on the head of the robot, determine whether or not the robot's gripper has successfully grasped the object.
Give your answer as YES or NO.
"""

# GRIP_IMG = cv2.imread('grip.png')

def gemini_init():
    # read your key from a text file. don't commit this! the repo is public.
    with open("api_key.txt", "r") as file:
        API_KEY = file.read()

    # Initialize Gemini API
    return genai.Client(api_key=API_KEY)

def prompt_gemini(client, mode, task="", img=None, loc=""):
    prompt = None
    if mode == "loc":
        prompt = LOC_PROMPT + task
    elif mode == "grip":
        # prompt = [GRIP_PROMPT, img]
        # Convert BGR → JPEG bytes
        _, buffer = cv2.imencode(".jpg", img)
        image_bytes = buffer.tobytes()
        img_clean = genai.types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        prompt = [GRIP_PROMPT, img_clean]
    elif mode == "renav":
        prompt = SEC_LOC_PROMPT + task + "\n" + loc
    else:
        print("Unrecognized mode. Aborting request.")
        return

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