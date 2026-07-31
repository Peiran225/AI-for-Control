# dreal/dreal4 publishes only linux/amd64.  Pin the immutable official image
# digest so the real Python binding cannot silently change between runs.
FROM dreal/dreal4:latest@sha256:6ec1d55b0eb5262dc9efb9f20e910ef997cee308ecad9d32a9a98ee0098b57d4

USER root

# The official image contains dReal's compiled Python 3.6 binding.  Keep that
# interpreter and use the last compatible scientific-stack releases.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates git python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -c "import dreal; x=dreal.Variable('x'); print(x*x+1)"

RUN python3 -m pip install --no-cache-dir --upgrade \
      pip==21.3.1 setuptools==59.6.0 wheel==0.37.1 \
    && python3 -m pip install --no-cache-dir \
      torch==1.10.2 \
      numpy==1.19.5 \
      scipy==1.5.4 \
      sympy==1.9 \
      matplotlib==3.3.4 \
      joblib==1.1.1 \
      tqdm==4.64.1 \
      pandas==1.1.5 \
      z3-solver==4.8.17.0

COPY external/lyznet /opt/ai-control/external/lyznet
RUN python3 -m pip install --no-cache-dir -e /opt/ai-control/external/lyznet

COPY faithful_related_work/lyznet/run_original_pinn_pi.py \
     /opt/ai-control/faithful_related_work/lyznet/run_original_pinn_pi.py

# Fail the image build if either the real binding or the author package is not
# importable.  This expressly forbids the legacy tumor stub workaround.
RUN python3 -c "import dreal,lyznet; assert hasattr(dreal,'CheckSatisfiability'); print(dreal.__file__)"

WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/ai-control/faithful_related_work/lyznet/run_original_pinn_pi.py"]
