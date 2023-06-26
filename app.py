from flask import Flask, request
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import os
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1000 * 1000
df_detector = load_model('df_detector.hdf5')
face_detector = mp.solutions.face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5)
input_shape = (256, 256, 3)

@app.route('/')
def hello():
    return 'hello'

@app.route('/predict', methods=['POST'])
# def predict():
#     faces = []
#     facePred = {}
#     b64_img = ""
#     try:
#       # {frame: [{id, x, y, w, h}, {...}, {...}, ...], ...}
#       # {id: 1/0, ...}
#         file = request.files['file']
#         filename = secure_filename(file.filename)
#         file.save(os.path.join('UPLOAD_FOLDER', filename))
#         img = cv2.imread(os.path.join('UPLOAD_FOLDER', filename))
#         gray = img[:,:,::-1]
#         img_height, img_width, _ = img.shape
#         facesDet = face_detector.process(gray)
#         i = 0
#         for face in facesDet.detections:
#             face_data = face.location_data.relative_bounding_box
#             x1 = int(face_data.xmin * img_width)
#             y1 = int(face_data.ymin * img_height)
#             width = int(face_data.width * img_width)
#             height = int(face_data.height * img_height)
#             x2 = int(x1 + width)
#             y2 = int(y1 + height)
#             crop_img = img[y1:y2, x1:x2]
#             resized_img = cv2.resize(crop_img, (256, 256)).reshape((1, 256, 256, 3))
#             pred = int(df_detector.predict(resized_img)[0][0])
#             id = filename + str(i)
#             newFace = {"id": id, "x": x1, "y": y1, "width": width, "height": height}
#             faces.append(newFace)
#             facePred[id] = pred
#             i += 1
#     except Exception as e:
#        print(e)
#        pass
#     for face in faces:
#         x1 = face["x"]
#         y1 = face["y"]
#         x2 = x1 + face["width"]
#         y2 = y1 + face["height"]
#         pred = facePred[face["id"]]
#         color = (0, 255, 0) if pred == 1 else (0, 0, 255)
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
#     retval, buffer = cv2.imencode(".png", img)
#     b64_img = str(base64.b64encode(buffer)).split('\'')[1]
#     os.remove(os.path.join('UPLOAD_FOLDER', filename))
#     data = {"faces": faces, "facePred": facePred, "base64": b64_img}
#     return data

def predict():
  # {frame: [{id, x, y, w, h}, {...}, {...}, ...], ...}
  # {id: 1/0, ...}
	prediction = 0
	count = 0
	faces = []
	facePred = {}
	b64_imgs = []
    
	try:
		def process_img(img, frameId):
			gray = img[:,:,::-1]
			img_height, img_width, _ = img.shape
			facesDet = face_detector.process(gray)
			temp = []
			i = 0
			try:
				for face in facesDet.detections:
					face_data = face.location_data.relative_bounding_box
					x1 = int(face_data.xmin * img_width)
					y1 = int(face_data.ymin * img_height)
					width = int(face_data.width * img_width)
					height = int(face_data.height * img_height)
					x2 = int(x1 + width)
					y2 = int(y1 + height)
					crop_img = img[y1:y2, x1:x2]
					resized_img = cv2.resize(crop_img, (256, 256)).reshape((1, 256, 256, 3))
					pred = int(df_detector.predict(resized_img)[0][0])
					id = filename + str(frameId) + str(i)
					newFace = {"id": id, "x": x1, "y": y1, "width": width, "height": height}
					temp.append(newFace)
					facePred[id] = pred
					i += 1
				for face in temp:
					x1 = face["x"]
					y1 = face["y"]
					x2 = x1 + face["width"]
					y2 = y1 + face["height"]
					pred = facePred[face["id"]]
					color = (0, 255, 0) if pred == 1 else (0, 0, 255)
					cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
					faces.append(temp)
				retval, buffer = cv2.imencode(".png", img)
				b64_img = str(base64.b64encode(buffer)).split('\'')[1]
				b64_imgs.append(b64_img)
			except Exception as e:
				print(e)
				pass

		file = request.files['file']
		filename = secure_filename(file.filename)
		file.save(os.path.join('UPLOAD_FOLDER', filename))
		file_type = file.content_type.split('/')[0]

		if(file_type == "image"):
			img = cv2.imread(os.path.join('UPLOAD_FOLDER', filename))
			process_img(img, 0)
		else:
			cap = cv2.VideoCapture(os.path.join('UPLOAD_FOLDER', filename))
			last_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
			frameRate = cap.get(5)
			while cap.isOpened():
				frameId = cap.get(1)
				ret, frame = cap.read()
				if frameId == last_frame_num:
					break
				if frameId % ((int(frameRate) + 1) * 1) == 0:
					process_img(frame, frameId)
			cap.release()

	except Exception as e:
		print(e)
		pass

	os.remove(os.path.join('UPLOAD_FOLDER', filename))
	count = len(faces)
	for each in facePred:
		prediction += facePred[each]
	if(count > 0):
		prediction = (prediction * 100.0) / (count * 1.0)
	else:
		prediction = ""
	data = {"base64": b64_imgs, "pred": prediction}
	return data

if __name__ == '__main__':
    app.run(host = '192.168.29.75', port = 5000, debug = True)