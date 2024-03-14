# STCANetVis-py
this the backend for STCANet Vis site it holds the model and server code

Open new terminal:
     Commands:
        1: cd Desktop
         2: mkdir scanet
         3: cd scanet

         4: git clone https://github.com/shabs76/STCANetVis-react.git
                  username: shabs76, password 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
         ikimaliza install  utaandika the following commands
          
          5: cd  STCANetVis-react
          6: npm install
           7: npm start

Ukimaliza those commands uI itakuwa active

Backend (model) 🥱

Open new terminal don't close uliyorun react
Commands:
     1: cd Desktop/scanet
      2: git clone https://github.com/shabs76/STCANetVis-py.git
             username: shabs76, password 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
       3: cd STCANetVis-py
       4: touch .env; cd db; touch .env; cd ..; code .
       kwenye vcode tafuta .env files the utapaste the following

        MYSQL_HOST='127.0.0.1'
        MYSQL_USER='scanetviz'
        MYSQL_PASSWORD='164G@DMADEITLVCR_FRV£@'
        MYSQL_APP_USER='scanapp'
        MYSQL_APP_USER_PASS='.5d$U7xF4Ww_F?+'
        MYSQL_DATABASE='theraplot'
        MYSQL_ROOT_PASS='164GDMADEITLVCR_FRV'
        MAIN_USER='theresa'

       4: cd db; docker compose -f docker-compose-local.yaml up
       Let me know kama kutawa na error
Open a new terminal  then:
       1: cd Desktop/scanet/STCANetVis-py; conda env create -f myenv.yml
wait for it to install everything
       3: conda activate
       4: cd Modal_Data
       5: python app.py
your sever itakuwa ready.

now open http://localhost:190

username: root
password: 164GDMADEITLVCR_FRV

on your left open click theraplot, then on top bar find import and click it
after that click choose file, nenda desktop, scanet then STCANetVis-py utaona file limeandikwa theresa.sql select hilo, scroll chini then utaclick import. Ukifika hapo text me