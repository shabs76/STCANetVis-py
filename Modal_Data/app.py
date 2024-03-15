import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
import math
# from flaskext.mysql import MySQL
import MySQLdb.cursors
import re
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
import string
import random
from flask_socketio import SocketIO, send, emit
import numpy as np
import json
import time
from flask_cors import CORS
# custom
import splitDataSet
import networkProcessor

UPLOAD_FOLDER = 'datasets'
ALLOWED_EXT = {'npy', 'csv'}

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001", "https://stcanetviz.brentles.co.tz"], supports_credentials=True)
mysql = MySQL(app)
bcrypt = Bcrypt(app)
socketio = SocketIO(app, cors_allowed_origins="*")

app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_APP_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_APP_USER_PASS')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DATABASE')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print(os.getenv('MYSQL_HOST'))
print(os.getenv('MYSQL_APP_USER'))
print(os.getenv('MYSQL_APP_USER_PASS'))
print(os.getenv('MYSQL_DATABASE'))
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


# MYSQL CONNECTION

# batches: 10,
# iteration: 30

PlayingUsers = {}

@socketio.on('connect')
def on_connect():
    socket_id = request.sid
    print(f"User connected with socket ID: {socket_id}")

@socketio.on('disconnect')
def on_disconnect():
    socket_id = request.sid
    if socket_id in PlayingUsers:
        PlayingUsers.pop(socket_id)
    print(f"User disconnected with socket ID: {socket_id}")

# socketio communication
@socketio.on('pause_model')
def handleDisc(disInf):
    userId = request.sid
    if userId in PlayingUsers:
        PlayingUsers.pop(userId)
    
    inff = {
        'state': 'success',
        'data': 'Model paused successfuly'
    }
    emit('pause_model', json.dumps(inff))
    print(f"User requested modal stop",{userId})

@socketio.on('play_model')
def handleModel(jsonData):
    socket_id = request.sid
    
    endRespo = {
        'state': 'end',
    }
    try:
        messsgae = json.loads(jsonData)
        print(messsgae)
        iniVa = {
            'state': 'starting',
            'data': 'model is starting now'
        }
        emit('play_model', json.dumps(iniVa))
        if 'passcode' not in messsgae:
            err = {
                'state': 'error',
                'data': 'Passcode is required'
            }
            emit('play_model', json.dumps(err))
            return json.dumps(endRespo)
        
        if 'project_id' not in messsgae:
            err = {
                'state': 'error',
                'data': 'Project Id is required'
            }
            emit('play_model', json.dumps(err))
            return json.dumps(endRespo)
    
        if 'iteration' not in messsgae:
            err = {
                'state': 'error',
                'data': 'Number of iterations is missing'
            }
            emit('play_model', json.dumps(err))
            return json.dumps(endRespo)
    
        if 'batches' not in messsgae:
            err = {
                'state': 'error',
                'data': 'Number of batches is missing'
            }
            emit('play_model', json.dumps(err))
            return json.dumps(endRespo)
        
        # now check if passcode is valid
        authAns = checkUserPassword(messsgae['passcode'])
        if authAns['state'] != 'success':
            emit('play_model', json.dumps(authAns))
            return json.dumps(endRespo)

        datasetAns = checkForDatasetDet(messsgae['project_id'])
        if datasetAns['state'] != 'success':
            emit('play_model', json.dumps(datasetAns))
            return json.dumps(endRespo)
        datasetName = datasetAns['data']['dataset_link']
        groupsAns = splitDataSet.load_csv_to_numpy('datasets/'+str(datasetName), messsgae['batches'])
        if groupsAns['state'] != 'success':
            emit('play_model', json.dumps(groupsAns))
            return json.dumps(endRespo)
        groups = groupsAns['data']
        topreAcc = random.uniform(3.40, 12.15)
        time.sleep(0.05)
        if isinstance(groups, list):
            PlayingUsers[socket_id] = socket_id #adding user to playing list
            TTrunTm = 0
            for group in groups:
                if socket_id not in PlayingUsers: #check if user is still playing the model and break they are not
                    break

                dataList = []
                predictions =[]
                ensembles = {
                    'mod1': [],
                    'mod2': [],
                    'x': group.shape[0]
                }
                epochBach = messsgae['iteration']//10
                startEpoch = epochBach
                while startEpoch <= messsgae['iteration']:
                    if socket_id not in PlayingUsers: #check if user is still playing the model and break they are not
                        break
                    shap = group.shape[0]
                    acc, predval, ens = networkProcessor.accuPro(shap, startEpoch, topreAcc)
                    if acc < 0:
                        acc = 0.001
                    runTim = networkProcessor.runTimePro(shap, startEpoch)
                    runDt = (acc, runTim, startEpoch, shap)
                    time.sleep(runTim/10000)
                    # if startEpoch == epochBach or messsgae['iteration'] == 100:
                    #     dataList.append(runDt)
                    dataList.append(runDt)
                    # prediction
                    predvalc = (shap+startEpoch, predval)
                    predictions.append(predvalc)
                    ensembles['mod1'].append(ens[0])
                    ensembles['mod2'].append(ens[1])
                    startEpoch += epochBach
                    TTrunTm += runTim
                mcm = networkProcessor.metrics(acc)
                sendPack = {
                    'state': 'success',
                    'data' : {
                        'predictions': predictions,
                        'len': groupsAns['wholeSh'],
                        'ensemble': ensembles,
                        'netData': dataList,
                        'mae': mcm['mae'],
                        'acc': acc,
                        'rmse': mcm['rmse'],
                        'epoch': 100,
                        'desire': round(topreAcc, 3)
                    }
                }

                emit('play_model', json.dumps(sendPack))
                mm = networkProcessor.metrics(acc)
                print(mm)
                print(mm['r_squared'])
            saveRunsD = recordDatasetRunInfo(
                mae=mm['mae'], r_square=mm['r_squared'], rmse=mm['rmse'], runtime=int(TTrunTm), run_csv='notset', epoch=100, dataset_id=messsgae['project_id'], nRows=groupsAns['wholeSh'])
            st = {
                'state': 'ending',
                'details': saveRunsD
            }
            emit('play_model', json.dumps(st))
            if socket_id in PlayingUsers:
                PlayingUsers.pop(socket_id)
        else:
            print(type(groups))
            err = {
                'state': 'error',
                'data': 'data format error'
            }
            emit('play_model', json.dumps(err))
    except Exception as e:
        print(e)
        err = {
            'state': 'error',
            'data': 'server error 01. Check logs'
        }
        emit('play_model', jsonify(err))
        return json.dumps(endRespo)
    return json.dumps(endRespo)





@app.route("/")
def intro():
    return "<p>TheraPlot APP API</p>"


@app.route("/dataupload", methods=['GET', 'POST'])
def datasetUpload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'dataset' not in request.files or 'project_name' not in request.form or 'passcode' not in request.form:
            err = {
                'state': 'error',
                'data': 'Missing file, passcode or project name'
            }
            return jsonify(err)
        file = request.files['dataset']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            err = {
                'state': 'error',
                'data': 'No selected file'
            }
            return jsonify(err)
        if file and allowed_file(file.filename):
            # check if project_name has valid values
            projectNa = request.form['project_name']
            if projectNa == '' or len(projectNa) < 5:
                err = {
                    'state': 'error',
                    'data': 'project name should have more than 5 characters'
                }
                return jsonify(err)
            psscheck = checkUserPassword(request.form['passcode'])
            if psscheck['state'] != 'success':
                return jsonify(psscheck)
            filename = secure_filename(file.filename)
            neName = generate_random_string(10)
            enam = os.path.splitext(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], neName+''+str(enam[1])))
            # dtlen = len(np.genfromtxt('datasets/'+neName+''+str(enam[1]),delimiter=',', skip_header=1, dtype=None, encoding=None))
            print('datasets/'+neName+''+str(enam[1]))
            # save to db
            suc = uploadDatasetsDB(filename, neName+''+str(enam[1]))
            return jsonify(suc)
        else:
            err = {
                'state': 'error',
                'data': 'File should be of npy type or csv'
            }
            return jsonify(err)
    else:
        err = {
            'state': 'error',
            'data': 'Not allowed'
        }
        return jsonify(err)

  
@app.route("/userCre", methods=['POST'])
def makeUser():
    if 'passcode' not in request.form:
        err = {
            'state': 'error',
            'data': 'missing passcode'
        }
        return jsonify(err)
    
    passcode = request.form['passcode']
    if passcode == '' or len(passcode) < 10:
        err = {
            'state': 'error',
            'data': 'passcode should more than 10 characters'
        }
        return jsonify(err)
    user_id = generate_random_string(12)
    hashPass = bcrypt.generate_password_hash(passcode).decode('utf-8')
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'INSERT INTO `users`(`user_id`, `passcode`, `user_name`, `date`) VALUES (%s,%s,%s, CURRENT_TIMESTAMP)',
        (
            user_id, hashPass, os.getenv('MAIN_USER')
        )
    )
    mysql.connection.commit()
    scu = {
        'state': 'success',
        'data': 'user was successfully added'
    }
    return jsonify(scu)


@app.route("/getinfo", methods=['POST'])
def getInfoApp():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    print(cursor)
    info = getAllModelsAndDetails()
    return jsonify(info)

def recordDatasetRunInfo(rmse, mae, r_square, epoch, runtime, run_csv, dataset_id, nRows):
    try:
        run_id = generate_random_string(12)
        qlIns = 'INSERT INTO `datasets_runs`(`run_id`, `rmse`, `mae`, `r_square`, `epoch`, `runtime`, `run_csv_path`, `num_rows`, `dataset_id`, `run_date`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,CURRENT_TIMESTAMP)' 
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            qlIns,
            (
                run_id, str(rmse), str(mae), str(r_square), epoch, runtime, run_csv, nRows,  dataset_id
            )
        )
        mysql.connection.commit()
        scu = {
            'state': 'success',
            'data': 'Model ran successfull and saved'
        }
        return scu
    except Exception as e:
        print(e)
        return {
            'state': 'error',
            'data': 'unable to save run details'
        }


def getAllModelsAndDetails():
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT `dataset_id`, `project_name`, `dataset_link`, `uploaded_date` FROM `datasets` WHERE 1 LIMIT %s ', (100,)
        )
        datasets = cursor.fetchall()
        if datasets:
            reDatasets = []
            for dataset in datasets:
                runSet = fetchRunsModel(dataset['dataset_id'])
                # { name: "Medi_sea", color: "blue", scalabilityInteraction: 0.185, scalabilityVisualization: 0.556, numbs: 12000 }
                print(dataset['dataset_id'])
                # calculate bubble data
                bubble = {}
                if runSet['state'] == 'success':
                    bubble = {
                    'name': dataset['project_name'],
                    'scalabilityInteraction': (int(runSet['data']['num_rows'])/10000)*0.134,
                    'scalabilityVisualization': (int(runSet['data']['num_rows'])/10000)*0.134*0.952,
                    'rows': int(runSet['data']['num_rows']),
                    'time': math.floor(int(runSet['data']['runtime'])/10000)
                }
                reDatasets.append({
                    'dataset': dataset,
                    'rundet': runSet,
                    'bubble': bubble
                })
            scu = {
                'state': 'success',
                'data': reDatasets
            }
            return scu
        else:
            err = {
                'state': 'error',
                'data': 'No dataset found'
            }
            return err 
    except Exception as e:
        print(e)
        return {
            'state': 'error',
            'data': 'Unable get data details'
        }

def fetchRunsModel(data_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT `run_id`, `rmse`, `mae`, `r_square`, `epoch`, `runtime`, `run_csv_path`, `num_rows`, `dataset_id`, `run_date` FROM `datasets_runs` WHERE dataset_id = %s  ORDER BY run_date DESC', (data_id,)
        )
        datasetsRuns = cursor.fetchone()
        if datasetsRuns:
            scu = {
                'state': 'success',
                'data': datasetsRuns
            }
            return scu
        else:
            err = {
                'state': 'error',
                'data': 'No run data found'
            }
            return err 
    except Exception as e:
        print(e)
        return {
            'state': 'error',
            'data': 'Unable get run de details'
        }


def uploadDatasetsDB(projectName, fileName):
    data_id = generate_random_string(12)
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'INSERT INTO `datasets`(`dataset_id`, `project_name`, `dataset_link`, `uploaded_date`) VALUES ( %s, %s,%s, CURRENT_TIMESTAMP)',
        (
            data_id, projectName, fileName
        )
    )
    mysql.connection.commit()
    scu = {
        'state': 'success',
        'data': {
            'info': 'Dataset was successfully uploaded and saved',
            'id': data_id
        }
    }
    return scu


def checkUserPassword(passcode):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'SELECT user_id, passcode, user_name, date FROM users WHERE user_name = %s ', (os.getenv('MAIN_USER'),)
    )
    account = cursor.fetchone()
    if account:
        validCode = bcrypt.check_password_hash(account['passcode'], passcode)
        if validCode:
            scu = {
                'state': 'success',
                'data': 'valid passcode'
            }
            return scu
        else:
            err = {
                'state': 'error',
                'data': 'invalid passcode'
            }
            return err
    else:
        err = {
            'state': 'error',
            'data': 'unable to get account response'
        }
        return err

def checkForDatasetDet(data_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'SELECT `dataset_id`, `project_name`, `dataset_link`, `uploaded_date` FROM `datasets` WHERE dataset_id = %s ', (data_id,)
    )
    dataset = cursor.fetchone()
    if dataset:
        scu = {
            'state': 'success',
            'data': dataset
        }
        return scu
    else:
        err = {
            'state': 'error',
            'data': 'Dataset is not found in the db'
        }
        return err


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string      


def load_npy_and_return_first_40_rows(file_path):
    try:
        data = np.load(file_path)
        first_40_rows = data[:100]
        suc = {
            'state': 'success',
            'data': first_40_rows.tolist()
        }
        return suc
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        err = {
            'state': 'error',
            'data': f"File '{file_path}' not found."
        }
        return err
    except Exception as e:
        print("An error occurred:", e)
        err = {
            'state': 'error',
            'data': 'An error occoured check logs for more info'
        }
        return err



if __name__ == '__main__':
    socketio.run(app, debug=True, port=8000)
