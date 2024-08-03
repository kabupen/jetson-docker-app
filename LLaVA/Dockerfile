FROM dustynv/pytorch:2.1-r36.2.0

RUN apt update && apt install -y vim

RUN git clone https://github.com/haotian-liu/LLaVA.git 
COPY pyproject.toml /LLaVA/
 
RUN cd /LLaVA && pip install -e .

RUN ln -s /usr/bin/python3 /usr/bin/python