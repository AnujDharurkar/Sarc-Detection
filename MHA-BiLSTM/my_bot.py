#Import Discord Package
import discord
import os
from SDTest import *

#Client (our bot)
client = discord.Client()

#Do Stuff
GUILD = os.getenv('DISCORD_GUILD')

client = discord.Client()


@client.event
async def on_ready():
    general_channel = client.get_channel(850356577445216289)
    await general_channel.send('Hello World!')

@client.event
async def on_message(message):
    print(message.author, client.user, type(client.user))
    if message.author == client.user:
        return

    addComment(message.content)
    predict()
    prediction = returnans()
    general_channel = client.get_channel(850356577445216289)
    if(prediction == 1):
        await general_channel.send("It's a Sarcastic Comment!")
    else:
        await general_channel.send("It's NOT a Sarcastic Comment!")

client.run('ODUwMzUyOTU3MTk0MTA4OTM4.YLoe5w.Ia4HWAuPdXf_9f6mryb5iPH2FEQ')