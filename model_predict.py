from keras.models import Sequential, model_from_json
from keras.layers import Dense
import zmq
import numpy as np

def get_prediction(X, model):
    prediction = model.predict(X)
    return(prediction)
    
    
print("Loading Model")
#load the model 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

print("Loaded model from disk")
print(model.summary())
model._make_predict_function()# have to initialize before threading
print("Done")


print("ZMQ Stuff...")
#open the sockets
# Initialize a zeromq context
context = zmq.Context()

# Set up a channel to receive work from the ventilator
work_receiver = context.socket(zmq.PULL)
work_receiver.connect("tcp://127.0.0.1:5560")

# Set up a channel to send result of work to the results reporter
results_sender = context.socket(zmq.PUSH)
results_sender.connect("tcp://127.0.0.1:5561")

#Set up a poller to to receive work
poller = zmq.Poller()
poller.register(work_receiver, zmq.POLLIN)
print("Done")
print("Waiting for work...")


try:
    while True:
        socks = dict(poller.poll())
        if socks.get(work_receiver) == zmq.POLLIN:
            work_message = work_receiver.recv_pyobj()
            Y = get_prediction(work_message, model)
            print(Y)
            #send the result back to the main program
            results_sender.send_pyobj(Y)
            
    
            
except KeyboardInterrupt:
    pass
