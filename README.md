![Contributors][contributors-shield]
![Forks][forks-shield]
![Stargazers][stars-shield]
![Issues][issues-shield]
[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/louisanhtran/)



<!-- Logo of website -->
<div align="center">

  <img src="https://effvision.com/wp-content/uploads/2024/06/artificial-intelligence-new-technology-science-futuristic-abstract-human-brain-ai-technology-cpu-central-processor-unit-chipset-big-data-machine-learning-cyber-mind-domination-generative-ai-scaled-1.jpg" width="500">

</div>

<!-- Introduction of project -->

<div align="center">
  
# 50.039 Theory and Practice of Deep Learning

</div>

<h2 align="center" style="text-decoration: none;">
Deep Learning Models for Time Series Data using Seq2Seq RNN and Transformer
</h2>

## About The Application :

Accurate weather forecasting plays a critical role in supporting urban planning, agriculture, public health, disaster preparedness, and transportation systems—particularly in highly populated cities like Delhi. However, the inherently dynamic and complex nature of weather patterns, influenced by both natural phenomena and human activities, makes precise forecasting a challenging task. In this project, we aim to improve the accuracy of short-term weather prediction in Delhi using deep learning. We train and evaluate multiple architectures, including a Seq2Seq RNN model, a Transformer-based model, and pre-trained time series models, on daily weather data collected between January 1, 2013, and April 24, 2017. The dataset includes four key features: average temperature, humidity, wind speed, and average air pressure. These models are used to predict future values of the four variables over the following months. Their performance is evaluated by comparing predictions to real recorded values, allowing us to assess model accuracy and robustness. The goal is to develop a reliable weather forecasting tool that can aid households, organizations, and government authorities in proactive decision-making and risk mitigation.

## Built With

This section outlines the technologies and tools used to develop the application.

* Backend: ![fastapi-shield][fastapi-shield]
* Frontend: ![fastapi-shield][streamlit-shield]
* AI/ML Framework: ![fastapi-shield][langchain-shield]
* LLM Provider: OpenAI
* Chat model: gpt-4o
* Embedding model: text-embedding-ada-002
* Vector database: Pinecone
* PDF Document storage: AWS S3 Bucket
* Caching: Redis
* Parsing large PDFs document: PyMuPDF

## Main components:

- API Endpoints are defined under [API Endpoints](https://github.com/LouisAnhTran/llm-powered-contextual-search-and-summarization-backend/blob/main/src/api/v1/app.py)
- Document indexing pipeline is defined under [Document Indexing Pipeline](https://github.com/LouisAnhTran/llm-powered-contextual-search-and-summarization-backend/blob/main/src/gen_ai/rag/doc_processing.py)
- Handling of LLM API calls and Chaining are managed and defined under [LLM API Calls + Chaining](https://github.com/LouisAnhTran/llm-powered-contextual-search-and-summarization-backend/blob/main/src/gen_ai/rag/chat_processing.py)
- Prompt templates are defined under [Prompt Templates](https://github.com/LouisAnhTran/llm-powered-contextual-search-and-summarization-backend/blob/main/src/gen_ai/rag/prompt_template.py)
- All constant variables needed to run the application are defined under [Configurations](https://github.com/LouisAnhTran/llm-powered-contextual-search-and-summarization-backend/blob/main/src/config.py)

## Application Demo:

[Watch the demo video on YouTube](https://www.youtube.com/watch?v=loZN4fdBfdU)


## Quick Application run using Docker

1. Install Docker:

  - Ensure Docker is installed on your machine. If not, download and install it from [Docker](https://docs.docker.com/engine/install/)

2. Create a Project Directory:
  - In your home directory, create a new directory for the project
    ```
    mkdir ai-app
    ```
  - Navigate to the newly created directory:
    ```
    cd ai-app
    ```
3. Clone the Project and Set Up Environment Variables

  - Clone the backend repository to your local machine:
     ```
     git clone https://github.com/LouisAnhTran/llm-powered-contextual-search-and-summarization-backend.git 
     ```
  - IMPORTANT! -> Navigate to the backend directory and paste the .env file containing all necessary credentials for the application.
  - Go back to the ai-app directory and clone the frontend repository:
     ```
     git clone https://github.com/LouisAnhTran/llm-powered-contextual-search-and-summarization-frontend.git
     ```
4. Create a docker-compose.yml File:
   - Create a docker-compose.yml file in ai-app directory with the following content:
     ```yml
     version: '3.8'

     services:
       redis:
         image: redis
         container_name: redis
         ports:
           - "6379:6379"
     
       backend:
         image: semantic-search-and-text-summarization-app-backend
         build:
           context: ./llm-powered-contextual-search-and-summarization-backend
           dockerfile: Dockerfile
         ports:
           - "8080:8080"
         env_file:
           - ./llm-powered-contextual-search-and-summarization-backend/.env
         depends_on:
           - redis
     
       frontend:
         image: semantic-search-and-text-summarization-app-frontend
         build:
           context: ./llm-powered-contextual-search-and-summarization-frontend
           dockerfile: Dockerfile
         ports:
           - "8501:8501"
         environment:
           - BACKEND_API_URL=http://backend:8080/api/v1
           - TENANT='staple_ai_client'
         depends_on:
           - backend

     ```

  - If you have followed all the steps correctly, your current directory should resemble the image below:

![Screenshot 2025-02-17 at 1 14 39 PM](https://github.com/user-attachments/assets/b31dbade-ddb4-44ea-a814-c4f09c5a3d0c)


    
  5. Run the application:
  - Run the following command to spin up three Docker containers (backend, frontend, and Redis server) to test the application:
   ```
    docker-compose up --build 
   ```
  - Wait for 10 seconds for the three containers to finish spinning up and running. Then, the application (frontend) should be running on port 8501. You can access it at [http://localhost:8501](http://localhost:8501)
  - To stop all running containers, press CTRL + C, then run:
  ```
    docker-compose down
  ```

<!-- GETTING STARTED -->
## Run application on Localhost (Without Docker)


### Prerequisites

1. Poetry:
  - Use the follow command to check whether poetry is installed in your machine
```
poetry --version
```
  - If poetry is not yet installed in your machine, please follow the link below to install Poetry, which is a common tool for dependency management and packaging in Python, specially for AI Applications.
[Poetry](https://python-poetry.org/docs/)

2. Docker:
  - Install Docker to run Redis server container later for caching LLM responses.
[Docker](https://docs.docker.com/engine/install/)

3. Clone the project to your local machine.

  ``` 
  git clone https://github.com/LouisAnhTran/llm-powered-contextual-search-and-summarization-backend.git
  ```


### Installation

1. Add the .env File:
- Place the .env file in the main project folder.
- It should contain all necessary credentials, including the AWS Access Key, OpenAI API Key, Pinecone API Key, etc.

2. Install Dependencies:
  ```sh
  poetry install
  ```

3. Create an Python virtual environment 

  ```sh
  poetry shell
  ```

4. Run the application
  ```sh
  poetry run python main.py   
  ```

5. Run the Redis server as a Docker container for caching LLM respone
- Open another terminal and type in following command
  ```sh
  docker run -d --name redis -p 6379:6379 redis      
  ```

6. Viewing API Endpoins through Swagger UI
- Right now the application should be up and running at port 8080. Click on [SwaggerUI](http://localhost:8080/docs) to view the list of all API endpoints.

7. Run Frontend
- Once Backend is alreaydy up and running, please access the following repo to set up and run the Frontend of the application. [Frontend Repo](https://github.com/LouisAnhTran/llm-powered-contextual-search-and-summarization-frontend)


## Architecture: 

### System Architecture:
   

![Screenshot 2025-02-17 at 1 48 22 PM](https://github.com/user-attachments/assets/c09edfc7-c2e2-45b9-aa64-86be6478cf76)


<!-- ACKNOWLEDGMENTS -->
## References:

- [FastAPI](https://fastapi.tiangolo.com/)

- [Redis](https://redis.io/)

- [Pipecone](https://www.pinecone.io/)

- [Streamlit](https://streamlit.io/)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png


[fastapi-shield]: https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi
[streamlit-shield]: https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white
[langchain-shield]: https://img.shields.io/badge/LangChain-ffffff?logo=langchain&logoColor=green




