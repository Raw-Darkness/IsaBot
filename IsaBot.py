from math import log
import discord
from openai import OpenAI
from collections import deque
import json
import requests
import io
import base64
import time
import logging
from logging.handlers import TimedRotatingFileHandler
from PIL import Image
import random
import asyncio
import workflow_api

# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a file handler to write logs to a file
file_handler = TimedRotatingFileHandler('app.log', when='midnight', interval=1, backupCount=7)
file_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger = logging.getLogger()
logger.addHandler(file_handler)


# Set an empty value to varible to avoid errors
last_image_generated_time = 0

#Path to config file
configPath = "Config.json"

# class for ratelimiting using a token bucket
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill_time = time.time()
        self.refill_rate = refill_rate

    def consume(self, tokens):
        current_time = time.time()
        self.tokens = min(self.capacity, self.tokens + (current_time - self.last_refill_time) * self.refill_rate)
        self.last_refill_time = current_time

        if tokens <= self.tokens:
            self.tokens -= tokens
            return True
        else:
            return False

# set token variables
bucket = TokenBucket(capacity=3, refill_rate=0.5)  # Example capacity and refill rate

# Open configuration file to get all custom values
try:
    with open(configPath, "r") as file:
        configFile = json.load(file)
    logging.info(f"Config file successfully opened with content: {configFile}")
except FileNotFoundError:
    logging.info("Config file not found. Please check the file path.")
except PermissionError:
    logging.info("Permission denied. Unable to open Config file.")
except Exception as e:
    logging.info("An error occurred:", str(e))

history_file_name = configFile["Name"] + "_message_histories.json"

# Load existing message histories from the JSON file when the bot starts
try:
    with open(history_file_name, 'r') as file:
        user_message_histories = json.load(file)
except FileNotFoundError:
    user_message_histories = {}

# set up variables for openai api calls
if configFile["OpenAPIKey"] is None:
    client = OpenAI(
    base_url=configFile["OpenAPIEndpoint"],
    )
else:
    client = OpenAI(
    base_url=configFile["OpenAPIEndpoint"],
    api_key=configFile["OpenAPIKey"],
    )

# Set up the deque to store message history
#message_history = deque(maxlen=6)

# Create a Discord bot instance
intents = discord.Intents.default()  # Create an instance of the Intents class
intents.message_content = True # Make sure message content is gathered
bot = discord.Client(intents=intents)  # Pass the intents argument during initialization

# ---------------------------- handling message events from discord
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return    
    """
    Check if the message is from an allowed channel or a direct message and not by an ignored user
    which is mainly other bots to avoid infinite bot discussion loops
    """
    
    # check if message is in allowed channels
    if message.channel.id in configFile["AllowedChannels"] or isinstance(message.channel, discord.DMChannel) and not message.author.id in configFile["IgnoredUsers"]:
        logging.info(f"Message recieved from discord: {message.author.name}: {message.content}")        
        
        # Check if bot only responds when called
        if configFile["OnlyWhenCalled"]:
            # Check if bot name is in message content or in DM channel or is mentioned
            if configFile["Name"].lower() in message.content.lower() or isinstance(message.channel, discord.DMChannel) or bot.user in message.mentions:
                # Check if draw is in message content
                if "draw" in message.content.lower():
                    await message.channel.send("Hang on while I sketch that for you...")
                    # Handle image generation
                    logging.info(f"Detected draw in text {message.content}")
                    await handle_image_generation(message)
                else:
                    # Handle message processing, removing the detected name from the prompt to avoid confusing the bot too much
                    message.content = message.content.replace(configFile["Name"].lower(), "")
                    await handle_message_processing(message)
            else:
                # Name was not in content, ignore message
                return
        # bot does not only reply when named, proceed with normal message handling
        else:
            if "draw" in message.content.lower():
                await message.channel.send("Hang on while I sketch that for you...")
                # Handle image generation
                logging.info(f"Detected draw in text {message.content}")
                await handle_image_generation(message)            
            else:
                # check against list of ignored words before acting
                if not any(word in message.content.lower() for word in configFile["IgnoredWords"]):                
                    # Handle message processing
                    await handle_message_processing(message)
                
                else:
                    # ignored word was in message
                    logging.info("Ignored word found in message, no reply will be generated")
                    return
    else:
        # Ignore messages from other channels or users
        return

# ---------------------------- handling generating images using comfyUI or automatic1111

def comfy_generate_image(comfy_prompt):
    logging.info("Sending request to Comfy API...")
    image_data = workflow_api.main(comfy_prompt)
    image = Image.open(io.BytesIO(image_data))
    logging.info("Image generated successfully.")
    return image    

def stable_diffusion_generate_image(prompt):
    try:
        # Make a POST request to the Stable Diffusion API
        response = requests.post(url=configFile["SDURL"], json={
            "prompt": configFile["SDPositivePrompt"] + prompt,
            "steps": configFile["SDSteps"],
            "width": configFile["SDWidth"],
            "height": configFile["SDHeight"],
            "negative_prompt": configFile["SDNegativePrompt"],
            "sampler_index": configFile["SDSampler"]
        })
        response.raise_for_status()
        logging.info("Stable Diffusion API call successful.")
        
        # Process the response to get the generated image
        r = response.json()
        image_data = base64.b64decode(r['images'][0])
        image = Image.open(io.BytesIO(image_data))
        logging.info("Image generated successfully.")
        return image
    except requests.exceptions.RequestException as e:
        logging.info("An error occurred during the Stable Diffusion API call:", str(e))
        return None

# ---------------------------- call correct image generating function, doing things such as throttling and detecting keywords

async def handle_image_generation(message):
    # check if draw requests are throttled or not  
    if bucket.consume(1):
        logging.info(f"Enough bucket tokens exist, running image generation")
        try:
            # Set the user's message as the prompt, removing the word "draw"
            prompt = message.content.replace("draw", "")
            if "--upscale" in prompt:
                # remove the command from the prompt
                prompt = message.content.replace("--upscale", "")
                # do logic for running upscaling of image
                # to be implemented

            logging.info(f"SD prompt by {message.author.name}, content is: {prompt}")        
            
            if configFile["UseComfy"]:
                image = comfy_generate_image(prompt)
            else:
                image = stable_diffusion_generate_image(prompt)
            """ # Generate the image using the stable diffusion API
            response = requests.post(url=configFile["SDURL"], json={
                "prompt": configFile["SDPositivePrompt"] + prompt,
                "steps": configFile["SDSteps"],
                "width": configFile["SDWidth"],
                "height": configFile["SDHeight"],
                "negative_prompt": configFile["SDNegativePrompt"],
                "sampler_index": configFile["SDSampler"]
            })
            r = response.json()
            image_data = base64.b64decode(r['images'][0])
            image = Image.open(io.BytesIO(image_data)) """

            # Wait for the image to be generated, also set status to typing        
            while not image_generated(image):            
                async with message.channel.typing():
                    await asyncio.sleep(1)     

            # Convert the image to bytes
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes.seek(0)

            # Create a discord.File object from the image bytes
            file = discord.File(image_bytes, filename='output.png')

            # Send the image file to the channel or direct message
            if isinstance(message.channel, discord.DMChannel):
                await message.author.send(file=file)
            else:
                await message.channel.send(file=file)
        except Exception as e:
            # Handle the error gracefully
            error_message = "An error occurred while generating the image. Please try again later."
            await message.channel.send(error_message)
            # Log the error for further investigation
            logging.info(f"Error occurred during image generation: {e}")
    else:
        await message.channel.send("Im busy sketching for you, please wait until I finish this one before asking for another.")
        logging.info("Image drawing throttled. Skipping draw request")
        return

# ---------------------------- format the correct llm prompt based on input data and send it to discord in dm or channel

async def handle_message_processing(message):
    try:       
        # append the message to the message history
        add_message_to_history('user', message.author.id, message.author.display_name, message.content)
        
        # Set the bot's status to "typing"
        async with message.channel.typing():
            # Process the user's message and generate a response using the OpenAI API
            response = generate_response(message.author.id)
        
        # Add the assistant's response to the message history
        add_message_to_history('assistant', message.author.id, message.author.display_name, response)
        
        # Send the response back to the user or direct message
        if isinstance(message.channel, discord.DMChannel):
            await message.author.send(response)
        else:
            await message.channel.send(response)
    except Exception as e:
        # Handle the error gracefully
        error_message = "Something went wrong, sorry!"
        await message.channel.send(error_message)
        # Log the error for further investigation
        logging.info(f"Error occurred during message processing: {e}")

# ---------------------------- handles adding items to the message history

def add_message_to_history(role, user_id, user_name, message_content):
        
    # if no memory set for that user, create one in the dictionary
    if user_id not in user_message_histories:
        user_message_histories[user_id] = []
        
    # append the message
    user_message_histories[user_id].append({'role': role, 'name': user_name,'content': message_content})
        
    # summarize the length of all the messages in the content field in the list of messages
    total_character_count = sum(len(entry['content']) for entry in user_message_histories[user_id])
    '''
    Check if the total character count exceeds 6000, roughly 2k tokens if average word is 3 characters long
    this is to make sure the totalt tokens doesnt exceed the 4k limit of the model
    we assume the personality promt takes rought 2k tokens as well
    if length is exceeded, remove oldes entry until we are below the limit
    '''
    logging.info(f'Total message history character count is: {total_character_count}')
    while total_character_count > 6000:        
        oldest_entry = user_message_histories[user_id].pop(0)
        total_character_count -= len(oldest_entry['content'])        
    
    # Save the updated message history to the JSON file
    try:
        with open(history_file_name, 'w') as file:
            json.dump(user_message_histories, file)
    except Exception as e:        
        logging.error(f"An error occurred while writing to the JSON file: {e}")

# ---------------------------- call the openai api to generate a text response

def generate_response(user_id):
    # Make a request to the OpenAI API to generate a response
    logging.info("Calling OpenAI API...")
    try:
        messages=[{'role': 'system', 'content': configFile["Personality"]}]
        for msg in list(user_message_histories[user_id]):
            if 'name' in msg:
                messages.append({'role': 'user', 'name': msg['name'], 'content': msg['content']})
            else:
                messages.append({'role': msg['role'], 'content': msg['content']})
        # remove the personality from the logging to avoid bloating it with the same text and only log the input prompt data
        logging.info(f"Sending data to openAI: {messages[1:]}")
        response = client.chat.completions.create(        
            messages=messages,
            model=configFile["OpenaiModel"]
        )
        logging.info("API response received.")

        # Extract the assistant's reply from the API response
        reply = response.choices[0].message.content

        return reply
    except Exception as e:
        # Log the error for further investigation
        logging.info(f"Error occurred during OpenAI API call: {e}")
        raise

# ---------------------------- simple function to wait image generation before continuing code

def image_generated(image):
    # Check if the image is valid
    if image is None:
        return False

    # Check if the image generation was successful    
    try:                
        # Return True if the image generation was successful, False otherwise
        return image.size[0] > 0 and image.size[1] > 0
    except Exception as e:
        # Handle any exceptions that occur during the image generation check
        # Return False if an error occurs
        logging.info(f"Error occurred during image generation: {e}")
        return False

# ---------------------------- run the bot

bot.run(configFile["DiscordToken"])