---
title: "Dialogflow Viber Connector"
excerpt: "Custom Viber messenger connector developed using Dialogflow V2 and Viber REST API in node.js"
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/Viber_Connector_Header.png
---

![Dialogflow_Viber_Connector](https://raw.githubusercontent.com/satishgunjal/images/master/Dialogflow_Viber_Connector.png)

In one of the production chatbot for a bank we were using the Dialogflow inbuilt connector for Viber integration. In the first quarter of 2020 Google decided to remove it from their platform. So in order to keep Viber channel functioning we have decided to create our own connector using Dialogflow API V2 and Viber REST API. We are using Dialogflow API to handle communication with Dialogflow and Viber API to handle the communication with Viber messenger. Since this is custom connector we have to also do the session management.

Since max lifespan of context in Dialogflow is 20 minutes, this also becomes max timeout for any inbuilt channel. In this connector we have kept the session timeout configurable and the customer can use any value from 0 to 20 minutes.

# GitHub Repository
* For more details and source code please refer [Viber-Connector-for-Dialogflow-Chatbot](https://github.com/satishgunjal/Viber-Connector-for-Dialogflow-Chatbot.git) of this project.
* Well documented code with detailed readme file will help you to test the application
  
# Role
* Architect & Project Manager

# Responsibilities
* Dialogflow and Viber API study
* Lead the project and support team

# Tech Stack
* Google Dialogflow Console
* Dialogflow & Viber REST API
* Node.js
* MS SQL Server
