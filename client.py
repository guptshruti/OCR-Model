import numpy as np
from PIL import Image
import requests
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

def client():
    ENABLE_SSL = True  # Update this based on your server's SSL configuration
    # ENDPOINT_URL = '20.244.7.112:8060'
    ENDPOINT_URL = "bhashini-iiith-bhasha-ocr.centralindia.inference.ml.azure.com"
    # HTTP_HEADERS = {"Authorization": "Bearer GOc667dOTqdLHD2mamT86ieVGiCzNDMh"} 
    #  # Replace with your authorization token
    HTTP_HEADERS = {"Authorization": "Bearer nZgOnCo5pYKru7Ps8B777CueXR1t9ipW"} 

    triton_http_client = httpclient.InferenceServerClient(
        url=ENDPOINT_URL, verbose=False, ssl=ENABLE_SSL
    )

    print("Is server ready - {}".format(triton_http_client.is_server_ready(headers=HTTP_HEADERS)))

    # image_url = "https://th.bing.com/th/id/OIP.IaeAoLbBvt_H0qqKlj89fAHaHa?rs=1&pid=ImgDetMain"
    # image_url = "https://dhruvacentrali0960249713.blob.core.windows.net/haridas/test1.jpg"
    # image_url = "https://th.bing.com/th/id/OIP.sJbwHJx_2thk6ltmowW_tQHaJi?rs=1&pid=ImgDetMain"
    image_url = '/home/azureuser/deployment_ocrmodel/data/English.jpg'
    language_code = "en"
    # image = np.asarray(Image.open(requests.get(image_url, stream=True).raw))
    image = np.asarray(Image.open(image_url).convert('RGB'))
    print("image_shape:", image.shape)
    input_language_id = np.array([language_code], dtype="object")

    input_tensors = [
        httpclient.InferInput("INPUT_IMAGE", image.shape, datatype=np_to_triton_dtype(image.dtype)),
        httpclient.InferInput("INPUT_LANGUAGE_ID", input_language_id.shape, datatype=np_to_triton_dtype(input_language_id.dtype))
    ]
    input_tensors[0].set_data_from_numpy(image)
    input_tensors[1].set_data_from_numpy(input_language_id)

    outputs = [httpclient.InferRequestedOutput("OUTPUT_TEXT")]
    model_name = "ocr"

    query_response = triton_http_client.infer(
        model_name=model_name, inputs=input_tensors, outputs=outputs, headers=HTTP_HEADERS
    )

    output = query_response.as_numpy("OUTPUT_TEXT")
    # print(output[0].decode())
    return output[0].decode()


import json
if __name__ == "__main__":
    output_text = client()
    # print("output_txt:", json.loads(output_text))
    for i in json.loads(output_text):
        print(i)
    # print("Output coordinates:", output_text)
