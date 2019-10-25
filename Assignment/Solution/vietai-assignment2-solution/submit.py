import requests
import json

name = input("Please input your name: ")

url = "http://35.240.129.160/judge/4/upload_file/"

files = {'file': ('activation_np.py', open('activation_np.py', 'rb'))}

print('Submitting activation_np.py')
response = requests.request("POST", url, files=files, data={'name': name})

response_data = json.loads(response.text)

print(response_data['score'])
print(response_data['message'])

url = "http://35.240.129.160/judge/5/upload_file/"

files = {'file': ('dnn_np.py', open('dnn_np.py', 'rb'))}

print('Submitting dnn_np.py')
response = requests.request("POST", url, files=files, data={'name': name})

response_data = json.loads(response.text)

print(response_data['score'])
print(response_data['message'])