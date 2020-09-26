---
title: "Banking Bot Published on Facebook and Viber"
excerpt: "Created this core banking digital assistance for leading bank in Bangladesh. Bank decided to make it publicaly avilable on Facebook and Viber Messenger"
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/Banking_Bot_Header1.png
---

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/EBL_DIA_Header.png)

This bot is based on 'Multimodal Bot solution' architecture. But since it's a project designed and developed based on customer specific requirements there are few changes in the architecture. Bot is developed using Dialogflow console and supports Facebook and Viber messenger channels. Here we are using inbuilt Dialogflow Facebook connector and since Viber connector is no more supported by the Google we have created customer Viber connector using Dialogflow and Viber API. We also have backend integrations with the customer provided REST API for core banking. Since customer provided web services returns the response withing 5 sec, we have kept the web hook integration inside the Dialogflow itself.

# Facebook Channel Architecture
Dialogflow and Facebook Messenger high level architecture is as below. Here we are using Dialogflow inbuilt Facebook Messenger connector.

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/EBL_DIA_Facebook.png)

## Salient Points
* Since we are using inbuilt Facebook Messenger connector, no programming is required here
* First create an app for a bot on Facebook developer site. Add necessary permission like 'pages_messaging'
* Get your Facebook Page Access Token and insert it in Dialogflow Facebook Messenger inbuilt connector
* We also need to create Verify Token in connector
* Use the Callback URL and Verify Token to create an event in the Facebook Messenger Web hook Setup.
* This will enable Facebook to send any message received in our bot, to Dialogflow for intent matching, entity extraction and generating the final response
* Dialogflow connector will send the response back to user using 'Facebook Messenger API'

#  Viber Channel Architecture
Dialogflow and Viber messenger high level architecture is as below. Here we are using custom Viber messenger connector.

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/EBL_DIA_Viber.png)

## Salient Points
* Since we are using custom-made Viber connector all the communication part is handled by our adapter. We just need Viber bot account and Dialogflow and Viber REST API for integration
* Viber connector is a node.js application with Dialogflow and Viber API integration.

### [Dialogflow API Integration](https://cloud.google.com/dialogflow/docs/reference/rest/v2-overview)

* We will use below Dialogflow API modules
  ```
  const dialogflow = require('dialogflow');
  ```
* Every message received through Viber API will be sent to Dialogflow to get the response
* Response received from Dialogflow is sent back to the user

### [Viber API Integration](https://developers.viber.com/docs/api/rest-bot-api/#message-types)

* We will be using below Viber API modules 
  ```
  const ViberBot = require('viber-bot').Bot;
  const BotEvents = require('viber-bot').Events;
  const TextMessage = require('viber-bot').Message.Text;
  const PictureMessage = require('viber-bot').Message.Picture;
  ```
* In order to receive the events from Viber messenger in our node.js connector we need to set up the web hook.
* Once you have your token you will be able to set your account’s web hook. This web hook will be used for receiving callbacks and user messages from Viber.
* Setting the web hook will be done by calling the set_webhook API with a valid & certified URL. This action defines the account’s web hook and the type of events the account wants to be notified about.
* For security reasons only URLs with valid and official SSL certificate from a trusted CA will be allowed. The certificate CA should be on the Sun Java trusted root certificates list.
* This will enable our application to receive the every message that user will type on Viber messenger bot

# Role
* Architect & product manager

# Responsibilities
* Requirement gathering
* Design and develop the chatbot for Facebook and Viber channel
* Lead the project and support team

# Tech Stack
* Google Dialogflow Console
* Dialogflow & Viber REST API
* Node.js
* MS SQL Server
