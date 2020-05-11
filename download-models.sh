################################################################################

# generic downloader / unpacker - (c) 2019 Toby Breckon, Durham University, UK

################################################################################

# *************** ICIP 2018 paper models - FireNet / InceptionV1-OnFire

################################################################################

URL=https://collections.durham.ac.uk/downloads/r19880vq98m
DIR_LOCAL_TARGET=models

FILE_NAME=dunnings-2018-fire-detection-pretrained-models.zip
DIR_NAME_UNZIPPED=dunnings-2018-fire-detection-pretrained-models
MD5_SUM=98815a8594a18f1cafb3e87af8f9b0f1

IDENTIFIER_STRING="ICIP 2018 Fire Detection CNN (FireNet / InceptionV1-OnFire) models"

UNCOMPRESS_COMMAND="unzip -q"

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

echo "Downloading $IDENTIFIER_STRING ..."

mkdir -p $DIR_LOCAL_TARGET

TARGET=./$DIR_LOCAL_TARGET/$FILE_NAME

curl -L -k $URL > $TARGET

################################################################################

# perform md5 check and move to required local target directory

cd $DIR_LOCAL_TARGET

echo "checking the MD5 checksum for downloaded $IDENTIFIER_STRING ..."

CHECK_SUM_CHECKPOINTS="$MD5_SUM  $FILE_NAME"

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the compressed file (using $UNCOMPRESS_COMMAND)..."

$UNCOMPRESS_COMMAND $FILE_NAME

echo "Tidying up..."

mv $DIR_NAME_UNZIPPED/* .

rm $FILE_NAME && rm -r $DIR_NAME_UNZIPPED

cd ..

################################################################################

# POST Download
# tlearn format specific - create checkpoint path files to enable conversion to pb format

echo "model_checkpoint_path: \"firenet\"" > $DIR_LOCAL_TARGET/FireNet/checkpoint
echo "all_model_checkpoint_paths: \"firenet\"" >> $DIR_LOCAL_TARGET/FireNet/checkpoint

echo "model_checkpoint_path: \"inceptiononv1onfire\"" > $DIR_LOCAL_TARGET/InceptionV1-OnFire/checkpoint
echo "all_model_checkpoint_paths: \"inceptiononv1onfire\"" >> $DIR_LOCAL_TARGET/InceptionV1-OnFire/checkpoint

echo "model_checkpoint_path: \"sp-inceptionv1onfire\"" > $DIR_LOCAL_TARGET/SP-InceptionV1-OnFire/checkpoint
echo "all_model_checkpoint_paths: \"sp-inceptionv1onfire\"" >> $DIR_LOCAL_TARGET/SP-InceptionV1-OnFire/checkpoint

################################################################################

echo "... completed -> required $IDENTIFIER_STRING are now in $DIR_LOCAL_TARGET/"

################################################################################

# ************ ICLMA 2019 paper models - InceptionV3-OnFire / InceptionV4-OnFire

echo; echo

################################################################################

URL=https://collections.durham.ac.uk/downloads/r25x21tf409
DIR_LOCAL_TARGET=models

FILE_NAME=samarth-2019-fire-detection-pretrained-models.zip
DIR_NAME_UNZIPPED=samarth-2019-fire-detection-pretrained-models
MD5_SUM=efa859a317ea0cb2ac27834662137500

IDENTIFIER_STRING="ICLMA 2019 Fire Detection CNN (InceptionV3-OnFire / InceptionV4-OnFire) models"

UNCOMPRESS_COMMAND="unzip -q"

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

echo "Downloading $IDENTIFIER_STRING ..."

mkdir -p $DIR_LOCAL_TARGET

TARGET=./$DIR_LOCAL_TARGET/$FILE_NAME

curl -L -k $URL > $TARGET

################################################################################

# perform md5 check and move to required local target directory

cd $DIR_LOCAL_TARGET

echo "checking the MD5 checksum for downloaded $IDENTIFIER_STRING ..."

CHECK_SUM_CHECKPOINTS="$MD5_SUM  $FILE_NAME"

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the compressed file (using $UNCOMPRESS_COMMAND)..."

$UNCOMPRESS_COMMAND $FILE_NAME

echo "Tidying up..."

mv $DIR_NAME_UNZIPPED/[SI]* .
echo >> README.txt
cat $DIR_NAME_UNZIPPED/README.txt >> README.txt

rm $FILE_NAME && rm -r $DIR_NAME_UNZIPPED

cd ..

################################################################################

# POST Download
# tlearn format specific - create checkpoint path files to enable conversion to pb format

echo "model_checkpoint_path: \"inceptiononv3onfire\"" > $DIR_LOCAL_TARGET/InceptionV3-OnFire/checkpoint
echo "all_model_checkpoint_paths: \"inceptiononv3onfire\"" >> $DIR_LOCAL_TARGET/InceptionV3-OnFire/checkpoint

echo "model_checkpoint_path: \"sp-inceptionv3onfire\"" > $DIR_LOCAL_TARGET/SP-InceptionV3-OnFire/checkpoint
echo "all_model_checkpoint_paths: \"sp-inceptionv3onfire\"" >> $DIR_LOCAL_TARGET/SP-InceptionV3-OnFire/checkpoint

echo "model_checkpoint_path: \"inceptiononv4onfire\"" > $DIR_LOCAL_TARGET/InceptionV4-OnFire/checkpoint
echo "all_model_checkpoint_paths: \"inceptiononv4onfire\"" >> $DIR_LOCAL_TARGET/InceptionV4-OnFire/checkpoint

echo "model_checkpoint_path: \"sp-inceptionv4onfire\"" > $DIR_LOCAL_TARGET/SP-InceptionV4-OnFire/checkpoint
echo "all_model_checkpoint_paths: \"sp-inceptionv4onfire\"" >> $DIR_LOCAL_TARGET/SP-InceptionV4-OnFire/checkpoint

################################################################################

echo "... completed -> required $IDENTIFIER_STRING are now in $DIR_LOCAL_TARGET/"

################################################################################
