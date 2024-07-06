from flask import Flask, request, jsonify
from firebase import firebase
url="https://for-test-4cb36-default-rtdb.firebaseio.com/"
fb=firebase.FirebaseApplication(url,None)

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username') # 使用者帳號
    password = data.get('password') # 使用者密碼
    role = data.get('role') #使用者身份

    user=tuple([username,password,role]) #使用者輸入

    datas=fb.get('/',None)
    users=set() #全部的人
    for data in datas.values():
        temp=[data['username'],data['password'],data['identity']]
        users.add(tuple(temp))
    if user in users:
        return jsonify({"status": "success", "message": "Login successful!", "role": role})
    else:
        return jsonify({"status": "error", "message": "Invalid username or password"})

    # 使用者的帳號密碼比對資料庫
    #if username == '123' and password == '123':
    #    if role in ['teacher', 'student']:
    #        return jsonify({"status": "success", "message": "Login successful!", "role": role})
    #    else:
    #        return jsonify({"status": "error", "message": "Invalid role"})
    #else:
    #    return jsonify({"status": "error", "message": "Invalid username or password"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
