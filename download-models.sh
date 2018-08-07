echo "Downloading pretrained models..."

mkdir models

MODELS=./models/checkpoints.zip
URL_MODELS=https://collections.durham.ac.uk/downloads/r19880vq98m

curl --progress-bar $URL_MODELS > $MODELS

cd models

echo "Unpacking the zip file..."

unzip -q checkpoints.zip

cp -R dunnings-2018-fire-detection-pretrained-models/. . 

rm checkpoints.zip && rm -r dunnings-2018-fire-detection-pretrained-models/

echo "All Done!!"


