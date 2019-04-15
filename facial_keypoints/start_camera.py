#!/usr/bin/python
# -*- coding=UTF-8 -*-

import cv2

import facial_keypoints_train as fk_train
import facial_keypoints_dataset as fk_dataset


class FaceCamera(object):

    def __init__(self,
                 kps_model_path='./saved_model/facial_keypoints_net.pt',
                 face_model_path='./haarcascade_frontalface_default.xml',
                 use_gpu_if_avaliable=False):

        print('Loading model ...')
        self.kps_model = fk_train.load_model(kps_model_path)
        self.kps_model.eval()
        self.face_model = cv2.CascadeClassifier(face_model_path)
        print('Loading done.')

        print('Preparing for capture...')
        self.cap = cv2.VideoCapture(0)
        print('Capture done.')

        self.rect = None
        self.all_kps = None

        self.image = None

        self.rect_expan = None

    def _handle(self):

        self.rect = self.face_model.detectMultiScale(self.image, 1.2, 2)
        self.all_kps = []

        for (x, y, w, h) in self.rect:
            face = self.image[y - self.rect_expan:y + h + self.rect_expan,
                   x - self.rect_expan:x + w + self.rect_expan]

            model_input = fk_dataset.transform_image(face)

            detected_kps = self.kps_model(model_input)
            detected_kps = detected_kps.view(68, 2)
            detected_kps = detected_kps.data.numpy()

            detected_kps = (detected_kps * 50.0) + 100
            detected_kps = detected_kps * (face.shape[1] / 224,
                                           face.shape[0] / 224)

            self.all_kps.append(detected_kps)

    def _draw(self):

        if self.rect is not None:

            for i, (x, y, w, h) in enumerate(self.rect):

                for point in self.all_kps[i]:
                    pt = (int(x + point[0] - self.rect_expan),
                          int(y + point[1] - self.rect_expan))
                    cv2.circle(self.image, pt, 1, (0, 0, 255), 4)

                cv2.rectangle(self.image, (x - self.rect_expan, y - self.rect_expan),
                              (x + w + self.rect_expan, y + h + self.rect_expan),
                              (255, 0, 0), 3)

    def start(self, handle_pre_frame=10,
              rect_expansion=60):

        self.rect_expan = rect_expansion

        count = 0
        while True:

            ret, frame = self.cap.read()

            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            count += 1
            if count % handle_pre_frame == 0:
                self._handle()

            self._draw()

            cv2.imshow("face camera", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera = FaceCamera()

    camera.start()
    camera.stop()
