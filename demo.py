import cv2, dlib, argparse
from contextlib import contextmanager
from utilities.dlib_utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image
from pathlib import Path
from tensorflow.keras.utils import get_file
from model import get_model
import numpy as np
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default="EfficientNetB0",
                        help="model name: 'ResNet50' or 'InceptionResNetV2' or 'EfficientNetB3")
    parser.add_argument("--weight_file", type=str, default="checkpoints/weights/dense256_finetune_unlocked-EfficientNetB0/512-0.001-adam/093-1.368-1.698.hdf5",
                        help="path to weight file (e.g. age_only_weights.029-4.027-5.250.hdf5)")
    parser.add_argument("--margin", type=float, default=1.1,
                        help="percentage of extra padding around face cropping box")
    parser.add_argument("--image_dir", type=str, default="../appa-real/test",
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    parser.add_argument("--image_gt", type=str, default="../appa-real/gt_test.csv",
                        help="ground truth labels for appa-real test; if set, visualization will render gt next to prediction")
    args = parser.parse_args()
    return args

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir, has_gt=False):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            if has_gt:
                yield cv2.resize(img, (int(w * r), int(h * r))), str(image_path).split("/")[3]
            else:
                yield cv2.resize(img, (int(w * r), int(h * r))), None

def main():
    args = get_args()
    model_name = args.model_name
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir
    image_gt = args.image_gt

    # for face detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("utilities/shape_predictor_68_face_landmarks.dat")

    # load model and weights
    model = get_model(model_name=model_name)
    model.load_weights(weight_file)
    img_size = model.input.shape.as_list()[1]

    # load ground truth labels for visualization
    gt = pd.read_csv(image_gt) if image_gt else None

    image_generator = yield_images_from_dir(image_dir, True if image_gt else False) if image_dir else yield_images()

    for img, path in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        if path:
            gt_slice = gt.loc[gt['file_name'] == path]
            boolean_series = gt["file_name"].str.startswith(path.split(".")[0])
            gt_slice = gt[boolean_series]
            apparent_age = gt_slice.apparent_age.mean()
            real_age = gt_slice.real_age.mean()
            print(gt_slice["apparent_age"], gt_slice["real_age"])
            print(apparent_age, real_age)
        
        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))
        face = None

        if len(detected) > 0:
            for i, d in enumerate(detected):
                shape = predictor(input_img, d)
                left_eye = extract_left_eye_center(shape)
                right_eye = extract_right_eye_center(shape)
                x1, y1, x2, y2, = d.left(), d.top(), d.right() + 1, d.bottom() + 1
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                M = get_rotation_matrix(left_eye, right_eye)
                rotated = cv2.warpAffine(img, M, (img_w, img_h), flags=cv2.INTER_CUBIC)
                
                face = faces[i, :, :, :] = cv2.resize(crop_image(rotated, d, margin), (224, 224))
                
                # x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                # xw1 = max(int(x1 - margin * w), 0)
                # yw1 = max(int(y1 - margin * h), 0)
                # xw2 = min(int(x2 + margin * w), img_w - 1)
                # yw2 = min(int(y2 + margin * h), img_h - 1)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                # # Debug line for looking at tightly cropped face
                # face = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results.dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                if path:
                    label = "P:" + str(int(predicted_ages[i])) + ", R:" + str(real_age) + ", A:" + str(round(apparent_age, 2))
                else:
                    label = "P:" + str(int(predicted_ages[i]))
                draw_label(img, (d.left(), d.top()), label)
                # draw_label(face, (0, 20), label, font_scale=0.7) # Temp line for tightly cropped face

        # cv2.imshow("result", img)
        try:
            cv2.imshow("result", img)
        except:
            print("imshow failed")

        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break


if __name__ == "__main__":
    main()
