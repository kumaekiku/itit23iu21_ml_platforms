# Prerequisite

Guide: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

After cloning this repo, run the following command to activate python virtual environment:

- Unix like system (linux):

```bash
# initialize python virtual env
python3.11 -m venv .env

# activate virtual env
source ./.env/bin/activate

# install required packages
pip install --upgrade pip
pip install -r ./requirements.txt
```

- Window same process but different approach so check guide link above, and make sure to run python version **3.11** (too lazy :D)
