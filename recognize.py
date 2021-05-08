import cv2
import numpy as np
import mtcnn
from architecture.resNetV2ModelStructure import *
from sklearn.preprocessing import Normalizer
from train import normalize
from scipy.spatial.distance import cosine
import pickle
import constants.variables as const
import constants.errors as errors


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


def detect(img, detector, encoder, encoding_dict, l2_normalizer):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < const.CONFIDENCE:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, const.REQUIRED_SIZE)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = ''

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < const.RECOGNITION and dist < distance:
                name = db_name
                distance = dist

        name = const.UNKNOWN if not name else name + f'  {distance:.2f}'
        cv2.rectangle(img, pt_1, pt_2, (80, 18, 236), 2)
        cv2.putText(img, name, (pt_1[0], pt_2[1] + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    return img


if __name__ == "__main__":
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights(const.FACENET_WEIGHTS)
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(const.EMBEDDINGS)
    l2_normalizer = Normalizer('l2')

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()

        if not ret:
            print(errors.CAMERA_ERROR)
            break

        frame = detect(frame, face_detector, face_encoder, encoding_dict, l2_normalizer)

        cv2.imshow(const.WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord(const.EXIT_VIDEO_KEY):
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
