################################################################################

# generic downloader / unpacker - (c) 2019 Toby Breckon, Durham University, UK

################################################################################

URL=https://collections.durham.ac.uk/downloads/r2d217qp536
DIR_LOCAL_TARGET=dataset

FILE_NAME=fire-dataset-dunnings-r2d217qp536-version1.zip
DIR_NAME_UNZIPPED=fire-dataset-dunnings
MD5_SUM=44c56cd3df8931c28b02910d35a4d105

IDENTIFIER_STRING="dataset"

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

echo "... completed -> required $IDENTIFIER_STRING is now in $DIR_LOCAL_TARGET/"

################################################################################
