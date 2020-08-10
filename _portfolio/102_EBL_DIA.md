---
title: "EBL DIA (Facebook, Viber)"
excerpt: "EBL DIA is the digital assistance created for EBL bank, and its vailable on Facebook messenger and Viber messenger"
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/EBL_DIA_Header.png
---

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/EBL_DIA_Header.png)

This bot is based on 'Multimodal Bot solution' architecture. But since its a project designed and developed based on customer specific requirements there few changes in the architecture. Bot is developed using Dialogflow console and supports Facebook messenger and Viber messenger channels. Here we are using inbuilt Dialogflow Facebook connector and since Viber conector is no more supported by the Google we have created customer Viber connector using DIalogflow and Viber API. We also have backend integrations with the customer provided REST web services. Since customer provided web services returns the response withing 5 sec, we have kept the webhook integration inside the Dialogflow itself.

# Dialofglow Facebook Messenger Architecture
Dialkoflow and Facebook messenger high level architecture is as below. Here we are using Dialogflow inbuilt Facebook messenger connector.

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/EBL_DIA_Facebook.png)

## Salient Points

# Dialofglow viber Messenger Architecture
Dialkoflow and Viber messenger high level architecture is as below. Here we are using custom Viber messenger connector.

![Multimodal_Bot_Solution_Header](https://raw.githubusercontent.com/satishgunjal/images/master/EBL_DIA_Viber.png)

## Salient Points

# Role
* Architect & product manager

# Responsibilities
* Requirement gathering
* Design and develop the chatbot for Facebook and Viber channel
* Lead the project and support team

# Tech Stack
* Google Dialogflow
* Node.js
* MS SQL Server
