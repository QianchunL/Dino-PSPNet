#!/bin/bash

# download PASCAL VOC 2012 dataset
#
# ref: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html

echo "Downloading..."

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar && \

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar && \

echo "Done."