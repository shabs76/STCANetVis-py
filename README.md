# STCANetVis-py
this is the backend for the STCANet Vis site, it holds the model and server code

## Requirements
+ Git
+ Docker
+ Docker compose

## Get Started
To get started with **STCANETVIS PYTHON** follow the following steps
+ Clone the repo
+ Get into the root of the project
  ```
  cd STCANetVis-py
  ```
+ Create _.env_ file and the following variables. MYSQL_HOST value should always be **theresia_maria_j** unless you change the name for the MariaDB container name.
  ```
  MYSQL_HOST='theresia_maria_j'
  MYSQL_USER='scanetviz'
  MYSQL_PASSWORD='normal_user_access_password'
  MYSQL_APP_USER='root'
  MYSQL_APP_USER_PASS='app_access_password'
  MYSQL_DATABASE='theraplot'
  MYSQL_ROOT_PASS='root_password'
  MAIN_USER='main_user'
  ```
+ Start all docker containers using the following command
  
  ```
  docker-compose up
  or
  docker compose up
  ```
+ Access Phpmyadmin from your browser using _localhost:190_
+ Login as root using the root password set in the .env file above
+ Import into the created database _MYSQL_DATABASE_ SQL file _theraplot.sql_ found in the root of the project
+ The app backend will be ready to be used.
+ To rebuild the application again run
  ```
  docker-compose up --build
  or
  docker compose up --build
