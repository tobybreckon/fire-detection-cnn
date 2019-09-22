################################################################################

# model downloader / unpacker - (c) 2018 Toby Breckon, Durham University, UK

################################################################################

URL_MODELS=https://collections.durham.ac.uk/downloads/r19880vq98m
MODEL_DIR_LOCAL_TARGET=models

MODELS_FILE_NAME=dunnings-2018-fire-detection-pretrained-models.zip
MODELS_DIR_NAME_UNZIPPED=dunnings-2018-fire-detection-pretrained-models
MODELS_MD5_SUM=98815a8594a18f1cafb3e87af8f9b0f1

################################################################################

# set this script to fail on error

set -e

# check for required commands to download and md5 check

(command -v curl | grep curl > /dev/null) ||
  (echo "Error: curl command not found, cannot download!")

(command -v md5sum | grep md5sum > /dev/null) ||
  (echo "Error: md5sum command not found, md5sum check will fail!")

################################################################################

# perform download

echo "Downloading pretrained models..."

mkdir -p $MODEL_DIR_LOCAL_TARGET

MODELS=./$MODEL_DIR_LOCAL_TARGET/$MODELS_FILE_NAME

curl -L -k $URL_MODELS > $MODELS

################################################################################

# perform md5 check and move to required local target directory

cd $MODEL_DIR_LOCAL_TARGET

echo "checking the MD5 checksum for downloaded models..."

CHECK_SUM_CHECKPOINTS="$MODELS_MD5_SUM  $MODELS_FILE_NAME"

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the zip file..."

unzip -q $MODELS_FILE_NAME

echo "Tidying up..."

mv $MODELS_DIR_NAME_UNZIPPED/* .

rm $MODELS_FILE_NAME && rm -r $MODELS_DIR_NAME_UNZIPPED

cd ..

################################################################################

# tlearn format specific - create checkpoint path files to enable conversion to pb format

echo "model_checkpoint_path: \"firenet\"" > $MODEL_DIR_LOCAL_TARGET/FireNet/checkpoint
echo "all_model_checkpoint_paths: \"firenet\"" >> $MODEL_DIR_LOCAL_TARGET/FireNet/checkpoint

echo "model_checkpoint_path: \"inceptiononv1onfire\"" > $MODEL_DIR_LOCAL_TARGET/InceptionV1-OnFire/checkpoint
echo "all_model_checkpoint_paths: \"inceptiononv1onfire\"" >> $MODEL_DIR_LOCAL_TARGET/InceptionV1-OnFire/checkpoint

echo "model_checkpoint_path: \"sp-inceptionv1onfire\"" > $MODEL_DIR_LOCAL_TARGET/SP-InceptionV1-OnFire/checkpoint
echo "all_model_checkpoint_paths: \"sp-inceptionv1onfire\"" >> $MODEL_DIR_LOCAL_TARGET/SP-InceptionV1-OnFire/checkpoint

################################################################################

echo "... completed -> required models are in $MODEL_DIR_LOCAL_TARGET/"

################################################################################
