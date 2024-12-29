echo -n 'Choose install option (1-Local 2-Docker): '
read -r option
if [ $option -eq 1 ]; then
    python3.11 -m venv .env
    source ./env/bin/activate
    pip install --upgrade pip
    pip install -r ./requirements.txt
    python3 ./train.py

elif [ $option -eq 2 ]; then
    echo 'Building docker image...'
    docker build -t pred_app .
    docker run -it --rm -p 8080:8080 pred_app
else
    echo "Invalid option... Terminate script."
    exit 1
fi
