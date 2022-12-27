FROM public.ecr.aws/lambda/python:3.9

RUN pip install keras_image_helper --no-cache-dir
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl?raw=true
#RUN pip3 install https://raw.githubusercontent.com/alexeygrigorev/serverless-deep-learning/master/tflite/tflite_runtime-2.2.0-cp37-cp37m-linux_x86_64.whl --no-cache-dir

COPY xray_model.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]
