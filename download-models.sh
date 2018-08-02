echo "Downloading pretrained models..."

mkdir Models

MODELS=./Models/checkpoints.zip
URL_MODELS=https://collections.durham.ac.uk/downloads/r19880vq98m

curl --progress-bar $URL_MODELS > $MODELS

echo "checking the MD5 checksum for downloaded models..."

cd Models

CHECK_SUM_CHECKPOINTS='dunnings-2018-fire-detection-pretrained-models  checkpoints.zip'

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the zip file..."

unzip -q checkpoints.zip

cp -R dunnings-2018-fire-detection-pretrained-models/. . 

rm checkpoints.zip && rm -r dunnings-2018-fire-detection-pretrained-models/

echo "All Done!!"


