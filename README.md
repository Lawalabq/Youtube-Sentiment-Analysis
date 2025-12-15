conda create -n venv python=3.11 -y

conda activate venv

pip install -r requirements.txt


##DVC   
dvc init

dvc repro

dvc dag

##AWS

aws configure