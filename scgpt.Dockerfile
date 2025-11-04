FROM xueerchen/scgpt:0.1.7

RUN pip install jupyter
Run pip install -U "scib-metrics==0.5.1" "jax[cuda12]" "chex==0.1.85"
