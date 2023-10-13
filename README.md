# Python Flask API for Philosophy Q&A

A modified and extended version of the [Encyclopedia-GPT repo](https://github.com/johngear/Encyclopedia-GPT) for API deployment in the cloud, quicker inferences, greater test coverage, written for a dedicated instance (as opposed to pure serverless backend). 

I have this code running on AWS EC2, using a Load Balancer to handle and route requests via my new private URL. 

The frontend that calls this is written in Svelte and hosted at https://philosophy-chat.com

[![Netlify Status](https://api.netlify.com/api/v1/badges/dbd6b317-8660-4d36-b96d-4245b95b4195/deploy-status)](https://app.netlify.com/sites/jrg/deploys)