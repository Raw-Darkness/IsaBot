# IsaBot
A simple python discord bot that uses OpenAI API's to generate LLM responses to user queries. The purpose is to make it feel like interacting with a human as much as possible. There are no commands required to use, simply chat away like with any user.

#### Disclaimer: 
The code of this bot has mostly been generated by chatGPT and I am by no means a good coder. It works but thats about it :)

![isabot](https://github.com/Raw-Darkness/IsaBot/assets/154023932/f5c4d753-59ef-46c8-bf02-7fd91fa53ddf)

### Features:
* Can use any OpenAI API to generate replies
* Can restrict access to specific channels, ignore specific users or messages containing certain words
* Can reply to all posts in a channel or only when mentioned, DM'ed or when name is detected in a message
* Can generate images when keyword is detected (draw by default) using automatic1111 api or comfyUI (not properly implemented due to their horrible API)
* Has a persistent memory for each user it interacts with that stays after reboots. Will be overwritten after a certain amount of characters are reached to prevent overreaching the token count
* Easily add a personality for the bot using the config file
* Multiple bots can be run simultaneously and can interact with eachtoher (risk of infinte loops though, be careful with this!), just run the script multiple times with different personalities and discord bot tokens.
* Rate limiting on image generation using a simple token system

### Steps to set up:
* Open config.json, fill in discord token, openAI API key, url for Automatic1111 or comfyUI, set other variables to your liking, such as personality, ignored users, allowed channels. <img width="1004" alt="image" src="https://github.com/Raw-Darkness/IsaBot/assets/154023932/5f7f4f2a-cf66-429a-80e0-6581a8b90071">
* Install requirements (pip install -r requirements.txt)
* Run IsaBot.py (python3 ./IsaBot.py)
* ???
* Profit!


### Config file:

    "DiscordToken": This should be your discord token, see https://www.wikihow.com/Create-a-Bot-in-Discord
    "OpenAPIKey": Add your api key from wherever you have the model, this could be OpenAI, OpenRouter or a locally hosted LLM in oogabooga
    "AllowedChannels": A list of channels the bot is allowed to post in. Only messages from these channels will be monitored and used to generate replies, se link below on how to get ID's (right click a channel to get the ID once you enable developer mode)
    "IgnoredUsers": a list of users that the bot will ignore messages from, see https://support.discord.com/hc/en-us/articles/206346498-Where-can-I-find-my-User-Server-Message-ID- for how to get the user ID's
    "IgnoredWords": if these words are detected in a message, that message will be ignored and no reply generated
    "OnlyWhenCalled": a boolean to determine if the bot only reacts when it detects its name in a message or to every message posted in allowed channels
    "OpenAPIEndpoint": The url to wherever you have your model
    "OpenaiModel": Which model you should use from the API endpoint, this depends on what models are available
    "SDURL": URL to automatic1111 API
    "SDPositivePrompt": these prompts will be added to whatever the user prompts for, can be left empty if you want the raw output from the SD prompt
    "SDNegativePrompt": Negative prompts added to the SD generation query
    "SDSteps": Amount of steps used when sending automatic1111 prompts
    "SDHeight": Resolution height for automatic1111 prompts
    "SDWidth": Resolution width for automatic1111 prompts
    "SDSampler": Sampler to use for automatic1111 prompts
    "ComfyURL":URL for comfy UI API, this does not currently work as comfyUI has a messy API
    "UseComfy": bool if comfy or automatic1111 should be used, currently only automatic1111 works so leave this as default
    "Name": Name of the bot
    "Personality": Personality of the bot. This will be added to all prompts to the LLM. Make sure not too make it too long as that will eat up the tokens of the model, leaving less for message history
