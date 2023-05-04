"""
This is a quickstart guide to using Steamship. It will show you how to
load a user-contributed package and invoke its AI.
"""
from steamship import Steamship

MY_ARTICLE = "Hello world! This is my article."

# Load a user-contributed package
api = Steamship.use("keyword-generator")

# Add AI to your app with a single line
keywords = api.invoke("generate", text=MY_ARTICLE)
print(keywords)
