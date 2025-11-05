FROM xueerchen/scgpt:0.1.7

RUN pip install jupyter
RUN pip install -U "llvmlite>=0.38.0,<0.39.0" "numba>=0.55.1,<0.56.0" "numpy<1.23"
RUN pip install --user matplotlib pandas