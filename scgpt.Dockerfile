FROM xueerchen/scgpt:0.1.7

# Optionnel : JupyterLab pour l'utilisation notebook (enlève si tu veux bosser en pur script/VS Code)
RUN pip install --quiet jupyterlab

# Forcer les versions strictes compatibles scGPT
RUN pip install --quiet "llvmlite>=0.38.0,<0.39.0" "numba>=0.55.1,<0.56.0" "numpy<1.23"

# Utile pour plots/données (souvent déjà dans l'image, mais ça ne fait pas de mal)
RUN pip install --quiet matplotlib pandas

# (PAS de scib-metrics, chex, jax ici : conflits, pas nécessaires pour le tuto d’annotation)
