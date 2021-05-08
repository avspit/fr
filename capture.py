import cv2
import os
import constants.variables as const
import constants.errors as errors

"""
Данный файл предназначен для записи короткого видео с лицом.
.   Для записи видео необходимо выполнить несколько шагов:
.       - задать имя в переменной face_name (в английской раскладке)
.       - запустить этот файл
.       - когда включится веб-камера, начнется запись видео
.       - поворачивайте голову в разных направлениях, для выхода нажмите 'q' на клавиатуре (запись будет остановлена, видео-файл сохранен)
"""

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # Получаем разрешение видео по-умолчанию
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Здесь задать имя перед тем, как записывать видео
    face_name = 'anton'

    face_dir = f'{const.FACES_TRAINING_DIR}{face_name}'
    os.makedirs(face_dir, exist_ok=True)

    # Определяем имя видео файла и видео-кодек
    out = cv2.VideoWriter(f'{face_dir}/{face_name}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print(errors.CAMERA_ERROR)
            break

        # Записываем кадр
        out.write(frame)

        cv2.imshow(const.WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()
