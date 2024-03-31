import onnxruntime as ort
import numpy as np
from PIL import Image

'''image_path = input('path: ')
image_path = image_path.strip('"')'''

b9 = [(0,0,113,113), (115,0,228,113), (231,0,344,113), 
      (0,115,113,228), (115,115,228,228), (231,115,344,228), 
      (0,231,113,344), (115,231,228,344), (231,231,344,344)]
p1 = (0,344,45,384)


def funcaptcha_3d(image):
    image = image.convert("RGB")

    model = ort.InferenceSession('siamese.onnx')
    def crop_funcaptcha_ans_image(image):
        return image.crop((0, 200, 135, 400))

    def process_pair_classifier_ans_image(image, input_shape=(52, 52)):
        sub_image = crop_funcaptcha_ans_image(image).resize(input_shape)
        return np.array(sub_image).transpose(2, 0, 1)[np.newaxis, ...] / 255.0

    def process_pair_classifier_image(image, index, input_shape=(52, 52)):
        x, y = index[1] * 200, index[0] * 200
        sub_image = image.crop((x, y, x + 200, y + 200)).resize(input_shape)
        return np.array(sub_image).transpose(2, 0, 1)[np.newaxis, ...] / 255.0

    def run_prediction(output_names, input_feed):
            return model.run(output_names, input_feed)


    max_prediction = float('-inf')
    max_index = -1

    width = image.width
    left = process_pair_classifier_ans_image(image)
    for i in range(width // 200):
        right = process_pair_classifier_image(image, (0, i))
        prediction = run_prediction(None, {'input_left': left.astype(np.float32),
                                                    'input_right': right.astype(np.float32)})[0]
        prediction_value = prediction[0][0]
        if prediction_value > max_prediction:
            max_prediction = prediction_value
            max_index = i



    return(max_index)








def test(image):
    image = image.convert("RGB")
    model = ort.InferenceSession('siamese.onnx')
    def crop_funcaptcha_ans_image(image, box):
        return image.crop(box)

    def process_pair_classifier_ans_image(image, box, input_shape=(32, 32)):
        sub_image = crop_funcaptcha_ans_image(image, box).resize(input_shape)
        return np.array(sub_image).transpose(2, 0, 1)[np.newaxis, ...] / 255.0
    
    def run_prediction(output_names, input_feed):
            return model.run(output_names, input_feed)
    

    left = process_pair_classifier_ans_image(image, p1)


    for i, box in enumerate(b9):
        aaa = image.crop(box)
        #aaa.show()
        right = process_pair_classifier_ans_image(image, box)

        prediction = run_prediction(None, {'input.1': left.astype(np.float32),
                                                    'input.53': right.astype(np.float32)})[0]
        prediction_value = prediction[0][0]

        print(f'图片{i}   概率{prediction_value}')

path = input('path: ')
i = Image.open(path)

test(i)