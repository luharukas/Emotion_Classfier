#!/bin/sh
sudo apt-get update
echo "***********************************Sudo Power initialized************"
sudo apt-get install python3.8 -y
echo "************************Python==3.8 version installed successfully***"
sudo apt install python3-pip -y
echo "************************************pip installation done************"
python3 --version
python3 -m pip --version
sudo apt-get install git -y
sudo apt-get install unzip -y
echo "******************************git installed*******************"
sudo apt install python3.8-venv
python3 -m venv ./my_pythonenv
echo "******************************virtual-env created********************"
source ./my_pythonenv/bin/activate
echo "**********************activation done********************************"

git clone https://github.com/luharukas/Face-Emotion-Recognition-as-Spatial-Image-using-Gabor-Filter.git
echo "*********************************Github repository cloned************"
cd Face-Emotion-Recognition-as-Spatial-Image-using-Gabor-Filter
pip install -r requirements.txt
unzip Data.zip
python3 main.py
streamlit run app.py