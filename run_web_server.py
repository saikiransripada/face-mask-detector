# import the necessary packages
import flask
from flask import request, render_template
import sqlite3

# initialize our Flask application and Redis server
app = flask.Flask(__name__)
# db = sqlite3.connect('test.db')

@app.route("/")
def home():
    try:
        conn = sqlite3.connect('sqlite.db')
        cursor = conn.cursor()
        print("Successfully Connected to SQLite")

        select = '''SELECT * FROM information'''
        cursor.execute(select)
        records = cursor.fetchall()

        conn.commit()
        print("Data inserted")

        cursor.close()

    except sqlite3.Error as error:
        print("Error while creating a sqlite table", error)
    finally:
        if (conn):
            conn.close()
            print("sqlite connection is closed")

    return render_template('index.html', records=records)


@app.route("/captures", methods=["POST"])
def predict():
    params = []
    params.append(request.form.get('name'))
    params.append(request.form.get('mask'))
    params.append(request.form.get('mask_probabilty'))
    params.append(request.form.get('face_probability'))
    params.append(request.form.get('filename'))
    params.append(request.form.get('captured_at'))

    try:
        conn = sqlite3.connect('sqlite.db')
        query = '''CREATE TABLE IF NOT EXISTS information (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            mask BOOLEAN NOT NULL,
            mask_probability FLOAT NOT NULL,
            face_probability FLOAT NOT NULL,
            filename TEXT NOT NULL,
            captured_at DATETIME NOT NULL,
            created_at DATETIME NOT NULL);'''

        cursor = conn.cursor()
        print("Successfully Connected to SQLite")
        cursor.execute(query)

        insert = '''INSERT INTO information (name,mask,mask_probability,face_probability,filename,captured_at,created_at)
            VALUES(?,?,?,?,?,?,datetime('now'))'''
        cursor.execute(insert, params)

        conn.commit()
        print("Data inserted")

        cursor.close()

    except sqlite3.Error as error:
        print("Error while creating a sqlite table", error)
    finally:
        if (conn):
            conn.close()
            print("sqlite connection is closed")

	# initialize the data dictionary that will be returned from the
	# view
    data = {"success": True}

	# return the data dictionary as a JSON response
    return flask.jsonify(data)


# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production
if __name__ == "__main__":
	print("* Starting web service...")
	app.run(debug=True)