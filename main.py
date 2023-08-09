import io
import json
import base64
from PIL import Image
import debugpy
import numpy as np

from ultralytics import YOLO

def init_context(context):
    context.logger.info("Init context... begin")
    # Load a model
    model = YOLO('/opt/nuclio/best.pt')
    context.user_data.model_handler = model

    debugpy.listen(5678)
    #Test 
    model(np.zeros((480, 640, 3), dtype=np.uint8))
    context.logger.info("Init context... end")

def handler(context, event):
    """Handles serverless request

    Args:
        context (_type_): _description_
        event (_type_): _description_
    """
    #From https://opencv.github.io/cvat/docs/manual/advanced/serverless-tutorial/#choose-a-dl-model
    context.logger.info("Run YOLOv8 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf)


    # Predict with the model
    pred_results = context.user_data.model_handler(image)[0]  # predict on an image

    # Prepare CVAT response
    results = []

    for result in pred_results.boxes.data:
        box = result.detach().cpu().numpy()
        results.append(
            {
                "confidence" : str(float(box[-2])),
                "label" : pred_results.names[int(box[-1])], 
                "points" : box[:4].tolist(),
                "type" : "rectangle"
            }
        )
    
    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)

