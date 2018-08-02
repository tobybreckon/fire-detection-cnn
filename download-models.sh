echo "Downloading pretrained models..."

mkdir Models

MODELS=./Models/checkpoints.zip
URL_MODELS=https://collections.durham.ac.uk/downloads/r19880vq98m

wget --quiet --show-progress $URL_MODELS -O $MODELS

echo "checking the MD5 checksum for downloaded models..."

cd models

CHECK_SUM_CHECKPOINTS='dunnings-2018-fire-detection-pretrained-models  checkpoints.zip'

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the zip file..."

unzip -q checkpoints.zip && rm checkpoints.zip && rm README.txt

echo "All Done!!"


