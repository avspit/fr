from architecture.resNetV2ModelStructure import *
import os
import cv2
import mtcnn
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
import constants.variables as const


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


if __name__ == "__main__":
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights(const.FACENET_WEIGHTS)
    face_detector = mtcnn.MTCNN()
    encodes = []
    encoding_dict = dict()
    l2_normalizer = Normalizer('l2')

    dirs = next(os.walk(const.FACES_TRAINING_DIR))[1]
    for label in dirs:
        for i, fn in enumerate(os.listdir(os.path.join(const.FACES_TRAINING_DIR, label))):
            print(f"Начинаем вычислять данные для лица {label}")
            cap = cv2.VideoCapture(os.path.join(const.FACES_TRAINING_DIR, label, fn))
            frame_count = 0
            while cap.isOpened():
                ret, raw_img = cap.read()
                # Обрабатываем каждый 5 кадр
                if frame_count % 5 == 0 and raw_img is not None:
                    #img_RGB = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                    img_RGB = raw_img
                    results = face_detector.detect_faces(img_RGB)

                    for res in results:
                        if res['confidence'] < const.CONFIDENCE:
                            continue
                        face, pt_1, pt_2 = get_face(img_RGB, res['box'])

                        cv2.imwrite(f'{const.FACES_TMP_DIR}{label}_{frame_count}.jpg', face)

                        face = normalize(face)
                        face = cv2.resize(face, const.REQUIRED_SIZE)
                        face_d = np.expand_dims(face, axis=0)
                        encode = face_encoder.predict(face_d)[0]
                        encodes.append(encode)

                        if encodes:
                            encode = np.sum(encodes, axis=0)
                            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
                            encoding_dict[label] = encode

                    with open(const.EMBEDDINGS, 'wb') as file:
                        pickle.dump(encoding_dict, file)

                frame_count += 1
                if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    break

            # Освобождаем ресурсы
            cap.release()
            cv2.destroyAllWindows()

        print("Выполнено")


