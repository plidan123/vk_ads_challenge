FROM continuumio/anaconda3:latest

CMD mkdir /opt/results
WORKDIR /opt/results
COPY baseline_always_zero.py baseline_always_zero.py

CMD python baseline_always_zero.py /tmp/data/test.tsv > /opt/results/result.tsv
