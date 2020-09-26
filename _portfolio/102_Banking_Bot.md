---
title: "Banking Bot Published on Facebook and Viber"
excerpt: "Created this core banking digital assistance for leading bank in Bangladesh. Bank decided to make it publically available on Facebook and Viber Messenger"
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/Banking_Bot_Header.png
---

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/EBL_DIA_Header.png)

This bot is based on 'Multimodal Bot solution' architecture. But since it's a project designed and developed based on customer specific requirements there are few changes in the architecture. Bot is developed using Dialogflow console and supports Facebook and Viber messenger channels. Here we are using inbuilt Dialogflow Facebook connector and since Viber connector is no more supported by the Google we have created customer Viber connector using Dialogflow and Viber API. We also have backend integrations with the customer provided REST API for core banking. Since customer provided web services returns the response withing 5 sec, we have kept the web hook integration inside the Dialogflow itself.

# Facebook Channel Architecture
Dialogflow and Facebook Messenger high level architecture is as below. Here we are using Dialogflow inbuilt Facebook Messenger connector.

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/EBL_DIA_Facebook.png)

## Salient Points
* Since we are using inbuilt Facebook Messenger connector, no programming is required here.
* First create an app for a bot on [Facebook developer](https://developers.facebook.com/) site. Add necessary permission like 'pages_messaging'.
* Get your Facebook Page Access Token and insert it in Dialogflow Facebook Messenger inbuilt connector.
* We also need to create Verify Token in connector.
* Use the Callback URL and Verify Token to create an event in the Facebook Messenger Web hook Setup.
* This will enable Facebook to send any message received in our bot, to Dialogflow for intent matching, entity extraction and generating the final response.
* Dialogflow connector will send the response back to user using 'Facebook Messenger API'

#  Viber Channel Architecture
Dialogflow and Viber messenger high level architecture is as below. Here we are using custom Viber messenger connector.

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/EBL_DIA_Viber.png)

## Salient Points
* Since we are using custom-made Viber connector all the communication part is handled by our adapter. We just need Viber bot account and Dialogflow and Viber REST API for integration
* Viber connector is a node.js application with Dialogflow and Viber API integration.
* For more details please refer my GitHub repository. [Viber-Connector-for-Dialogflow-Chatbot](https://github.com/satishgunjal/Viber-Connector-for-Dialogflow-Chatbot.git)

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
