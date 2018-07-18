echo "Downloading pretrained models..."

mkdir models

MODELS=./models/checkpoints.zip
URL_MODELS=https://community.dur.ac.uk/atharva.s.deshmukh/test/Localisation_Demo.mp4

wget --quiet --show-progress $URL_MODELS -O $MODELS

# echo "checking the MD5 checksum for downloaded models..."

cd models

# CHECK_SUM_CHECKPOINTS='b176b00450ce9aaf3ef812087ed3ef49  checkpoints.zip'

# echo $CHECK_SUM_CHECKPOINTS | md5sum -c

# echo "Unpacking the zip file..."

# unzip -q checkpoints.zip && rm checkpoints.zip && rm README.txt

echo "All Done!!"


