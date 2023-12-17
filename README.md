IsaBot
A simple python discord bot that uses OpenAI API's to generate LLM responses to user queries.
Features:
* Can use any OpenAI API to generate replies
* Can restrict access to specific channels, ignore specific users or messages containing certain words
* Can reply to all posts in a channel or only when mentioned, DM'ed or when name is detected in a message
* Can generate images when keyword is detected (draw by default) using automatic1111 api or comfyUI (not properly implemented due to their horrible API)
* Has a persistent memory for each user it interacts with that stays after reboots. Will be overwritten after a certain amount of characters are reached to prevent overreaching the token count
* Easily add a personality for the bot using the config file
* Multiple bots can be run simultaneously and can interact with eachtoher (risk of infinte loops though, be careful with this!), just run the script multiple times with different personalities and discord bot tokens.
* Rate limiting on image generation using a simple token system

  Steps to set up:
  * Open config.json, fill in discord token, openAI API key, url for Automatic1111 or comfyUI, set other variables to your liking, such as personality, ignored users, allowed channels.
  * Run IsaBot.py
  * ???
  * Profit!