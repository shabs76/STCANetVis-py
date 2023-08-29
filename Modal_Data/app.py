import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
import string
import random
from flask_socketio import SocketIO, send, emit
import numpy as np
import json

UPLOAD_FOLDER = 'datasets'
ALLOWED_EXT = {'npy'}

load_dotenv()

app = Flask(__name__)
mysql = MySQL(app)
bcrypt = Bcrypt(app)
socketio = SocketIO(app)

app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_APP_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_APP_USER_PASS')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DATABASE')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT



# socketio communication

@socketio.on('play_model', namespace='/model')
def handleModel(jsonData):
    try:
        messsgae = json.loads(jsonData)
        print(messsgae)
        emit('play_model', json.dumps(messsgae))
        if not messsgae['passcode']:
            err = {
                'state': 'error',
                'data': 'Passcode is required'
            }
            emit('play_model', json.dumps(err))
        # now check if passcode is valid
        authAns = checkUserPassword(messsgae['passcode'])
        if authAns['state'] != 'success':
            emit('play_model', json.dumps(authAns))
        # now call data function
        datInf = load_npy_and_return_first_40_rows('datasets/u.npy')
        # print(datInf)
        nun = 1
        # emit('play_model', {'data': nun})
        for nn in datInf['data']:
            # print(nn)
            for n in nn:
                # print(n)
                for jj in n:
                    print(jj)
                    print('break')
                    print('break')
                    print('break')
                    emit('play_model', {'data': jj})
                    nun = nun + 1
    except Exception as e:
        print(e)
        err = {
            'state': 'error',
            'data': 'server error 01. Check logs'
        }
        emit('play_model', jsonify(err))



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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # save to db
            suc = uploadDatasetsDB(projectNa, filename)
            return jsonify(suc)
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
        'data': 'Dataset was successfully uploaded and saved'
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
    socketio.run(app, debug=True)