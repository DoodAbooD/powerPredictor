from http.server import HTTPServer, BaseHTTPRequestHandler
from predictor import *
import json
import torch
import pandas as pd
import numpy as np


class predictHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		self.send_response(200)
		self.send_header('content-type' , 'text/html')
		self.end_headers()
		self.wfile.write("Power Predictor by Abdelqader & Mohammad,use POST method to get predictions".encode())
	def do_POST(self):
		self.send_response(200)
		self.send_header('content-type' , 'application/json')
		self.end_headers()
		#print(self.headers['content-length'])

		content_length = int(self.headers['Content-Length'])
		post_data = self.rfile.read(content_length)
		data_string = post_data.decode("UTF-8")
		data_dict = json.loads(data_string)
		#print(data_dict[0]["temp"])
		
		#Passing input to prediction model
		function = data_dict["func"]
		if (function == "p"):
			inputTensor1 = torch.tensor([	float(data_dict["data"][0]["temp"]) ,
							float(data_dict["data"][0]["hum"]) ,
						 	float(data_dict["data"][0]["time"]) ])
			inputTensor2 = torch.tensor([	float(data_dict["data"][1]["temp"]) ,
							float(data_dict["data"][1]["hum"]) ,
						 	float(data_dict["data"][1]["time"]) ])
			inputTensor3 = torch.tensor([	float(data_dict["data"][2]["temp"]) ,
							float(data_dict["data"][2]["hum"]) ,
						 	float(data_dict["data"][2]["time"]) ])
			inputTensor4 = torch.tensor([	float(data_dict["data"][3]["temp"]) ,
							float(data_dict["data"][3]["hum"]) ,
						 	float(data_dict["data"][3]["time"]) ])
			inputTensor5 = torch.tensor([	float(data_dict["data"][4]["temp"]) ,
							float(data_dict["data"][4]["hum"]) ,
						 	float(data_dict["data"][4]["time"]) ])
		
			
			with torch.no_grad():
				model_output1 = model(inputTensor1).item()
				model_output2 = model(inputTensor2).item()
				model_output3 = model(inputTensor3).item()
				model_output4 = model(inputTensor4).item()
				model_output5 = model(inputTensor5).item()			
			print ("First Tensors Received:")
			print (inputTensor1)

			energy_prediction = (model_output1 + model_output2 + model_output3 + model_output4 + model_output5)*13/5 
			energy_prediction = "{:2f}".format(energy_prediction)
			post_output = str.encode('{"energy":' + '"' + str(energy_prediction) + '"' + '}')
			#post_output = str.encode(energy_prediction)
			print ("Sending: ")
			print(post_output)
			self.wfile.write(post_output)

		elif (function == "d"):
			data_time = float(data_dict["data"][0]["time"])
			data_temp = float(data_dict["data"][0]["temp"])
			data_hum = float(data_dict["data"][0]["hum"])
			data_power = float(data_dict["data"][0]["power"])
			data_frame = pd.DataFrame(np.array([[data_time,data_temp,data_hum,data_power]], dtype=np.float32))
			data_frame.to_csv('dataReal.csv', index=False, header=False, mode='a')			
			post_output = str.encode(f'Entry: ({data_time} {data_temp} {data_hum} {data_power}) was added to the dataset')
			print("Sending: ")
			print(post_output)
			self.wfile.write(post_output)

		elif (function == "i"):
			data_MSE = MSE
			data_csv = pd.read_csv('dataReal.csv')
			data_count = len(data_csv.index)
			post_output = str.encode('{"MSE":' + '"' + str(data_MSE) + '"' + ',' + '"count":"' + str(data_count) + '"}')
			print("Sending: ")
			print(post_output)
			self.wfile.write(post_output)



def main():
	print(input_size)
	PORT = 7895
	server = HTTPServer(('', PORT), predictHandler)
	print("Server running on port ", PORT)
	server.serve_forever()


if __name__ == '__main__':
	main()

