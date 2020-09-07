---
title: "Multimodal Bot Solution"
excerpt: "Multimodal bot solution uses bot framework like Google Dialogflow and custom designed adapters to publish bot across variety of channels"
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/Multimodal_Bot_Solution_Header.png
---

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/Multimodal_Bot_Solution_Header.png)


Multimodal bot solution is designed from ground up to utilize bot framework like Google Dialogflow and create text and speech based conversational AI interfaces across variety of channels. Strategy was to start with simple directive dialog chatbot and build on to it to create more complex conversational flows. My IVR experience came handy to create state machine design pattern for chatbot design. Because of following reasons we finalized the Google Dialogflow NLU platform,
* Dialogflow is an end-to-end, build-once deploy-everywhere development suite for creating conversational interfaces for websites, mobile applications, popular messaging platforms, and IoT devices.
* You can use it to build interfaces (such as chatbots and conversational IVR) that enable natural and rich interactions between your users and your business.
* It works on Google Cloud Platform, so its scalable and supports pay-as-use billing.

Designing the conversational experience was the one of the major challenge we faced while working on conversational bot development. We ended up creating our own design pattern, which uses the traditional state machine approach along with dialogflow specific features like context to add memory to bot and increase the intent matching accuracy.

Once we were able to develop the conversational bot we started building our own custom connectors for end channels. We used Dialogflow API V2 to create custom connectors. We have created the custom connectors for Viber and WhatsApp channel. This was the most important milestone, as it eliminated our dependency on Dialogflow provided connectors. There are few inherent disadvantages with dialogflow inbuilt  connectors like no control over session management, 5 seconds web hook timeout limit and limited channel specific events.

So once we gain the expertise in dialogflow API, we modified our connectors to handle channel session management, web hook management and adding channel specific custom events.
This made our multimodal bot solution foolproof and created solid base for building feature rich bots for our customers. 

The way we have designed our solution, we are only using the Dialogflow NLU engine for intent matching and entity extraction whereas session management and web hook management is handled by our custom connectors. If required we can also replace the Dialogflow NLU with some other NLU if required. I am now working on TensorFlow Deep Learning framework and NLP.

# Multimodal Bot Architecture
High level architecture of multimodal bot solution is as below

![Multimodal_Bot_Solution_Architecture](https://raw.githubusercontent.com/satishgunjal/images/master/Multimodal_Bot_Solution_Architecture.png)

## Salient Points 
* This can be implemented on premise or on cloud
* User can use speech or text to communicate with the bot.
* Our platform supports almost all the possible channels like website bot, social medial channels, Google assistance and even smart speakers
* We can use reverse proxy for load balancing and routing the channel specific requests
* Create separate connector for each channel.
* Workaround for 5 seconds web hook timeout
* Unlimited channel specific events
* Each adapter will use dialogflow API for communication with dialogflow agent(intent matching & entity extraction) and channel specific API like Viber adapter will use Viber REST API.
* Bot works on request response basic. Each adapter will keep listening for any communication from the user, these messages come in the form of events. Adapter will parse the response from the events and send the user input to Dialogflow agent. Response from the Dialogflow agent will be sent back to the user using 'send API'. 
* If required, bot adapter can log the custom business logic related data for reporting purpose. 
* We can also add the agent transfer functionality by using socket.io or WebRTC protocols.

# Role
* Architect & product owner
* Individual contributor and manager

# Responsibilities
* Design and develop the chatbot solution and lead the team
* Lead the chatbot projects and POC
* Partner collaboration
* Responding to RFP

# Tech Stack
* Google Dialogflow
* Eudata
* Node.js
* Python
* Machine Learning
