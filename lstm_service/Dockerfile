FROM public.ecr.aws/lambda/python:3.12
# CPU-only PyTorch keeps the image a few hundred MB instead of several GB
RUN pip install --no-cache-dir numpy && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY forecast.py handler.py ${LAMBDA_TASK_ROOT}/
CMD ["handler.handler"]
