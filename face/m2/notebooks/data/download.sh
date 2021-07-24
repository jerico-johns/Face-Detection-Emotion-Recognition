# +
wget -nc https://berkeley-w251-jjonte.s3.amazonaws.com/frame_images_DB.tar.gz
tar -xf frame_images_DB.tar.gz --skip-old-files

wget -nc https://berkeley-w251-jjonte.s3.amazonaws.com/aligned_images_DB.tar.gz
tar -xf aligned_images_DB.tar.gz --skip-old-files

wget -nc https://berkeley-w251-jjonte.s3.amazonaws.com/archive.zip
unzip archive.zip -d ./celebs
