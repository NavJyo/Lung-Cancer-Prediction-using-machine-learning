Lung Cancer Prediction:

Table of Content
1. Demo
2. Overview
3. Motivation
4. Installation
5. Deployement on Heroku
6. Directory Tree
7. Bug / Feature Request

Demo

Link: https://pcos26.herokuapp.com 


Overview

This is a simple Flask web app which predicts whether a patient is having Lung Cancer or not.  
![PCOS](https://user-images.githubusercontent.com/36689965/117628129-715c0980-b196-11eb-86c0-2aea49d1948a.JPG)


Motivation

With the amount of new diseases coming up every day, there is a need for an effective method to diagnose diseases.  This was one of the disease prediction used for my B.tech major project. 

Installation

The Code is written in Python 3.6.10. If you don't have Python installed you can find it [here](https://www.python.org) . If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:


pip install -r requirements.txt

Deployement on Heroku

Login or signup in order to create virtual app. You can either connect your github profile or download ctl to manually deploy this project.
Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python)  to deploy a web app.


Directory Tree

├── static 

 │   ├── css
 
├── template

 │   ├── lung.html
 
├── Procfile

├── README.md

├── app.py 

├── lungcancer.pkl

├── requirements.txt
 

Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue here by including your search query and the expected result 

Technologies Used


![flask](https://user-images.githubusercontent.com/36689965/117626347-a23b3f00-b194-11eb-8d75-222752930976.png)
![sklearn](https://user-images.githubusercontent.com/36689965/117563487-e1e62600-b0c3-11eb-83bb-e6cb104408f2.png)
![1_yX4UOWqDDStHTRHgCbUyvA](https://user-images.githubusercontent.com/36689965/117629368-b896ca00-b197-11eb-89f5-0810af23c700.png)
