---
title: "Multimodal Bot Solution"
excerpt: "Multimodal bot solution uses bot framework like Google Dialogflow and custom designed adapters to publish bot across variety of channels"
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/Multiodal_Bot_Solution_Teaser_421x410.png
---

Multimodal bot solution is designed from gound up to utilize bot framework like Google dialogflow and create text and speech based conversation AI interface across variety of channels. Statergy was to start with very basic form of chatbot like FQA bot and utilize existing framework to create it. Once we gain the experties to create directive dialog chatbot we started working on conversatiobal AI chatbots. We finalized the Google dialogflow framework as our main bot frameworks and created the conversational bots.

Designing the conversational experience was the one of the major challenge, we faced while working on conversational bot development. We ended up creating our own design pattern, which uses the traditional statemachine approach along with dialogflow specific features like context to add mnemory to bot and increase the intent matching accuracy.

One we were able to develope the conversational bot we have decided to start building our own connector for end channels. We have created the custom adapters for Viber and WhatsApp channel. This was most important milestone as it eliminated our dependancy on dialogflow provided channels. When we use inbuilt dialoflow connectors then we loose the control over the session management, webhook timeout, handling channel specific events.

So once we gain the expertise in dialohflow API, we modifed our adapters to handle channel session management, webhook management and addding channel specific custom events.
This made our bot applications fulproof and created solid base for building feature rich bots for our customers. 

I am now working on final part of this statergy where I would like to create our own NLU engine using opensource API like RASA to handle the intent matching and entity extraction. Since we are already gain the expertise to handle channel session management, external webhook management and event management we need our NLU engine to do intent matching and entity extraction only. This is still work in progress.

# Role & Responsibilities
* Architect
* Design and develope the chatbot solution and lead the team
* Lead the chat bot projects and POC
* Partner collaboration

# Multimodal Bot Architecture
High level architecture of multimodal bot solution is as below

![Multimodal_Bot_Solution_Architecture](https://raw.githubusercontent.com/satishgunjal/images/master/Multimodal_Bot_Solution_Architecture.png)

* This can be imlemented on premise or on cloud
* User can use speech or text to comminicate with the bot.
* Our platform supports almost all the possible channels like website bot, social medial channels, Google assitance and even smart speakers
* We can use reverse proxy for 



