import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# model = load_model('mp_hand_gesture')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

def getClasses(gesture_file):
	f = open('gesture.names', 'r')
	cn = f.read().split('\n')
	f.close()
	return cn

classNames = getClasses("gesture.names")


actions = np.array(classNames)

def get_label(index, hand, results):
	output = None
	for idx, classification in enumerate(results.multi_handedness):
		if classification.classification[0].index == index:
			label = classification.classification[0].label
			output = label
	return output


def extract_keypoints(result):
	if result.multi_hand_landmarks:
		if len(result.multi_hand_landmarks) == 1:
			hand = result.multi_hand_landmarks[0]
			hand_label = get_label(1, hand, result)
			if hand_label == 'Left':
				left_hand = np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten() 
				right_hand = np.zeros(21*3)
			else:
				right_hand = np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten() 
				left_hand = np.zeros(21*3)
		else:
			hand1 = result.multi_hand_landmarks[0]
			hand2 = result.multi_hand_landmarks[1]
			hand_label = get_label(1, hand1, result)
			if hand_label == 'Left':
				left_hand = np.array([[res.x, res.y, res.z] for res in hand1.landmark]).flatten() 
				right_hand = np.array([[res.x, res.y, res.z] for res in hand2.landmark]).flatten() 
			else:
				right_hand = np.array([[res.x, res.y, res.z] for res in hand1.landmark]).flatten() 
				left_hand = np.array([[res.x, res.y, res.z] for res in hand2.landmark]).flatten() 
	else:
		left_hand = np.zeros(21*3)
		right_hand = np.zeros(21*3)
	landmark = np.concatenate([left_hand,right_hand])
	return landmark



def mediapipe_detection(image, model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False
	results = model.process(image)
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image, results


def draw_hand_landmarks(image, result):
	landmarks = []
	if result.multi_hand_landmarks:
		for handslms in result.multi_hand_landmarks:
			for lm3 in handslms.landmark:
				landmarks.append([lm3.x, lm3.y, lm3.z])
			mp_drawing.draw_landmarks(image, handslms, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(255,255,255), thickness=1,circle_radius=1),mp_drawing.DrawingSpec(color=(255,51,255), thickness=1,circle_radius=1))


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,126) ))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# model = keras.models.load_model("ABC_model")
model.load_weights('Good_model')

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
	output_frame = input_frame.copy()
	# for num, prob in enumerate(res):
	# 	print(num, prob)
	# 	cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
	# 	cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)    
	return output_frame


sequence = []
sentence = []
threshold = 0.8
predictions = []


cap = cv2.VideoCapture(0)
# Set mediapipe model 
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
# breakval = False
# while cap.isOpened():
# 	ret, frame2 = cap.read()
# 	if (ret == True) and (not breakval):
# 		frame = frame2
# 		# Make detections
# 		image, results = mediapipe_detection(frame, hands)

# 		draw_hand_landmarks(image, results)

# 		keypoints = extract_keypoints(results)
# 		# print(len(keypoints))
# #         sequence.insert(0,keypoints)
# #         sequence = sequence[:30]
# 		sequence.append(keypoints)
# 		sequence = sequence[-30:]

# 		if len(sequence) == 30:
# 			res = model.predict(np.expand_dims(sequence, axis=0))[0]
# 			print(actions[np.argmax(res)])


#         #3. Viz logic
# 			if res[np.argmax(res)] > threshold: 
# 				if len(sentence) > 0: 
# 					if actions[np.argmax(res)] != sentence[-1]:
# 						sentence.append(str(actions[np.argmax(res)]) + " - " + str(res[np.argmax(res)])) 
# 				else:
# 					sentence.append(str(actions[np.argmax(res)]) + " - " + str(res[np.argmax(res)])) 

# 			if len(sentence) > 1: 
# 				sentence = sentence[-1:]

#             # Viz probabilities
# 			# image = prob_viz(res, actions, image, colors)

# 		cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
# 		cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# 		# Show to screen
# 		cv2.imshow('OpenCV Feed', image)

# 		# Break gracefully
# 		if cv2.waitKey(1) & 0xFF == ord('q'):
# 			breakval = True
# 			break
# 	else:
# 		break
# cap.release()
# cv2.destroyAllWindows()
# cv2.waitKey(1)
valid = True
while cap.isOpened():

	# Read feed
	ret, frame = cap.read()

	# Make detections
	image, results = mediapipe_detection(frame, hands)
	# print(results)

	# Draw landmarks
	draw_hand_landmarks(image, results)

	# 2. Prediction logic
	keypoints = extract_keypoints(results)
	sequence.append(keypoints)
	
	# print(len(sequence))
	if len(sequence) >= 12:
		sequence = sequence[-10:]
		res = model.predict(np.expand_dims(sequence, axis=0))[0]
		# print(res)
		# prints label
		print(actions[np.argmax(res)])
		predictions.append(np.argmax(res))
		sentence.append(actions[np.argmax(res)])

		# 3. Viz logic
		# prints prob
		print(res[np.argmax(res)])
		# prints label map val
		# print(np.unique(predictions[-2:])[0])
		# print(np.argmax(res))
		valid = False
		if np.unique(predictions[-4:])[0]==np.argmax(res): 
			if res[np.argmax(res)] > threshold: 
				valid = True
				if len(sentence) > 0: 
					if actions[np.argmax(res)] != sentence[-1]:
						sentence.append(actions[np.argmax(res)])
						# cv2.waitKey(500)
				# else:
					# sentence.append(actions[np.argmax(res)])
		# # print(sentence)
		if len(sentence) > 1: 
			sentence = sentence[-1:]

		# Viz probabilities
		# image = prob_viz(res, actions, image, colors)
	    
	cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
	if valid: 
		cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

	# Show to screen
	cv2.imshow('OpenCV Feed', image)

	# Break gracefully
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

