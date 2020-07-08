import os
import sys
import optparse
import subprocess
import random
import numpy as np
import keras
import datetime
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model


try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
PORT = 8873
import traci


class DQNAgent_tri:
    def __init__(self):
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.1  # exploration rate
        self.learning_rate = 0.0002
        self.memory = deque(maxlen=200)
        self.model = self._build_model()
        self.action_size = 2

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(9, 9, 1))
        x1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_1)
        x1 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x1)
        x1 = Flatten()(x1)

        input_2 = Input(shape=(9, 9, 1))
        x2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_2)
        x2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x2)
        x2 = Flatten()(x2)

        input_3 = Input(shape=(2, 1))
        x3 = Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2, x3])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        model.compile(optimizer=keras.optimizers.RMSprop(
            lr=self.learning_rate), loss='mse')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
'N10'

class SumoTrisection:
    def __init__(self,junction,lanes):
        self.junction = junction
        self.lanes = lanes

    def calculateReward(self, wgMatrix):
        reward = 0
        r2 = 0
        r3 = 0
        r4 = 0
        vehicles_road1 = traci.edge.getLastStepVehicleIDs(str(self.lanes[0]))
        vehicles_road2 = traci.edge.getLastStepVehicleIDs(str(self.lanes[1]))
        vehicles_road3 = traci.edge.getLastStepVehicleIDs(str(self.lanes[2]))

        for v in vehicles_road1:
            if (len(weight_matrix) != 0 and v in wgMatrix):
                 #reward += wgMatrix[v]
                reward += (wgMatrix[v]*traci.vehicle.getWaitingTime(self,v))
                r4 += wgMatrix[v]

            else:
                reward = reward + 1
                r4 = r4 + 1

        for v in vehicles_road2:
            if (len(weight_matrix) != 0 and v in wgMatrix):
                #reward += wgMatrix[v]
                reward += (wgMatrix[v]*traci.vehicle.getWaitingTime(self,v))
                r3 += wgMatrix[v]
            else:
                reward = reward + 1
                r3 = r3 + 1

        for v in vehicles_road3:
            if (len(weight_matrix) != 0 and v in wgMatrix):
                #reward += wgMatrix[v]
                reward += (wgMatrix[v]*traci.vehicle.getWaitingTime(self,v))
                r2 += wgMatrix[v]
            else:
                reward = reward + 1
                r2  = r2 + 1



        return [reward, r2, r3, r4]



       
    def getState(self):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 7
        offset = 11
        speedLimit = 14

        junctionPosition = traci.junction.getPosition(str(self.junction))[0]
        vehicles_road1 = traci.edge.getLastStepVehicleIDs(str(self.lanes[0]))
        vehicles_road2 = traci.edge.getLastStepVehicleIDs(str(self.lanes[1]))
        vehicles_road3 = traci.edge.getLastStepVehicleIDs(str(self.lanes[2]))
        for i in range(9):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(9):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)

        for v in vehicles_road1:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
            if(ind < 9):
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][8 - ind] = 1
                velocityMatrix[2 - traci.vehicle.getLaneIndex(
                    v)][8 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road2:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
            if(ind < 9):
                positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition(str(self.junction))[1]
        for v in vehicles_road3:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
            if(ind < 9):
                positionMatrix[6 + 2 -
                               traci.vehicle.getLaneIndex(v)][8- ind] = 1
                velocityMatrix[6 + 2 - traci.vehicle.getLaneIndex(
                    v)][8 - ind] = traci.vehicle.getSpeed(v) / speedLimit


        light = [0, 1]

        position = np.array(positionMatrix)
        position = position.reshape(1, 9, 9, 1)

        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 9, 9, 1)

        lgts = np.array(light)
        lgts = lgts.reshape(1, 2, 1)

        return [position, velocity, lgts]
