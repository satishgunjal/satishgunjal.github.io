---
title: "Multimodal Bot Solution"
excerpt: "Multimodal bot solution uses bot framework like Google Dialogflow and custom designed adapters to publish bot across variety of channels"
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/Multimodal_Bot_Solution_Header.png
---

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/Multimodal_Bot_Solution_Header.png)

Multimodal bot solution is designed from ground up to utilize bot framework like Google dialogflow and create text and speech based conversation AI interface across variety of channels. Strategy was to start with directive dialog chatbot and FAQ bot and utilize existing framework to create it. Once we gain the expertise to create directive dialog chatbot we started working on conversational AI chatbots. We finalized the Google dialogflow framework as our main bot framework and created the conversational bots.

Designing the conversational experience was the one of the major challenge we faced while working on conversational bot development. We ended up creating our own design pattern, which uses the traditional state machine approach along with dialogflow specific features like context to add memory to bot and increase the intent matching accuracy.

Once we were able to develop the conversational bot we started building our own customer adapters for end channels. We have created the custom adapters for Viber and WhatsApp channel. This was the most important milestone, as it eliminated our dependency on dialogflow provided connectors. There are few inherent disadvantages with dialogflow inbuilt  connectors like no control over session management, 5 sec web hook timeout limit and limited channel specific events.

So once we gain the expertise in dialogflow API, we modified our adapters to handle channel session management, web hook management and adding channel specific custom events.
This made our bot applications foolproof and created solid base for building feature rich bots for our customers. 

I am now working on final part of this strategy where I would like to create our own NLU engine using open source API like RASA to handle the intent matching and entity extraction. Since we have already gained the expertise to handle channel session management, external web hook and event management we need our NLU engine to do intent matching and entity extraction only. This is still work in progress.

# Role & Responsibilities
* Architect
* Design and develop the chatbot solution and lead the team
* Lead the chatbot projects and POC
* Partner collaboration

# Multimodal Bot Architecture
High level architecture of multimodal bot solution is as below

![Multimodal_Bot_Solution_Architecture](https://raw.githubusercontent.com/satishgunjal/images/master/Multimodal_Bot_Solution_Architecture.png)

## Salient Points 
* This can be implemented on premise or on cloud
* User can use speech or text to communicate with the bot.
* Our platform supports almost all the possible channels like website bot, social medial channels, Google assistance and even smart speakers
* We can use reverse proxy for load balancing and routing the channel specific requests
* Create separate adapter for each channel.
* Each adapter will use dialogflow API for communication with dialogflow agent(intent matching & entity extraction) and channel specific API like Viber adapter will use Viber REST API.
* Bot works on request response basic. Each adapter will keep listening for any communication from the user, these messages come in the form of events. Adapter will parse the response from the events and send the user input to dialogflow agent. Response from the dialogflow agent will be sent back to the user using 'send API'. 
* If required, bot adapter can log the custom business logic related data for reporting purpose. 
* We can also add the agent transfer functionality by using socket.io or WebRTC protocols.
