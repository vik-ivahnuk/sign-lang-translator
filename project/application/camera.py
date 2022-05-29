import numpy as np
import onnxruntime as ort
import cv2


class SignLangTranslator:
    alphabet = list('ABCDEFGHIKLMNOPQRSTUVWXY')

    ort_ = ort.InferenceSession("model.onnx")
    winname = "Sign Language Translator"

    mean = 0.485 * 255.
    std = 0.229 * 255.

    @staticmethod
    def __camera_center(frame):
        height, width, _ = frame.shape
        start = abs(height - width) // 2
        if height > width:
            frame = frame[start: start + width]
        else:
            frame = frame[:, start: start + height]
        return frame

    def start_translate(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        i = 0
        text = ''
        while True:
            i = i + 1
            ret, frame = cap.read()
            frame = SignLangTranslator.__camera_center(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            x = cv2.resize(frame, (28, 28))
            x = (x - self.mean) / self.std

            x = x.reshape(1, 1, 28, 28).astype(np.float32)
            y = self.ort_.run(None, {'input': x})[0]

            index = np.argmax(y, axis=1)
            letter = self.alphabet[int(index)]
            if i % 10 == 0:
                i = 0
                text += letter
            cv2.putText(frame, letter, (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("Sign Language Translator", frame)

            k = cv2.waitKey(1)
            if cv2.getWindowProperty(self.winname, cv2.WND_PROP_AUTOSIZE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()
        return text


if __name__ == '__main__':
    sign_translator = SignLangTranslator()
    sign_translator.start_translate()