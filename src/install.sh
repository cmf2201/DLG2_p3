#!/bin/bash

echo "###########################"
echo "Preparing Guo's network ..."
tar xf ./repository/master.tar.gz -C ./repository/

mkdir -p ./models/guo
cp ./repository/Learning-Monocular-Depth-by-Stereo-master/models/model_utils.py ./models/guo/
cp ./repository/Learning-Monocular-Depth-by-Stereo-master/models/monocular_model.py ./models/guo/
cp ./repository/Learning-Monocular-Depth-by-Stereo-master/list/eigen_test_list.txt ./Src/list/
cp ./repository/Learning-Monocular-Depth-by-Stereo-master/list/eigen_train_list.txt ./Src/list/
cp ./repository/Learning-Monocular-Depth-by-Stereo-master/list/eigen_val_list.txt ./Src/list/

# modify files
sed -i -e "s/from\ model_utils\ import\ \*/from\ \.model_utils\ import\ \*/g" ./models/guo/monocular_model.py
sed -i -e "s/from\ utils\.util_functions\ import\ unsqueeze_dim0_tensor/#from\ utils\.util_functions\ import\ unsqueeze_dim0_tensor/g" ./models/guo/monocular_model.py
sed -i -e "s/print(\"loading\ pretrained\ weights\ downloaded\ from\ pytorch.org\")/#print(\"loading\ pretrained\ weights\ downloaded\ from\ pytorch.org\")/g" ./models/guo/monocular_model.py
sed -i -e "s/print(\"do\ not\ load\ pretrained\ weights\ for\ the\ monocular\ model\")/#print(\"do\ not\ load\ pretrained\ weights\ for\ the\ monocular\ model\")\n            pass/g" ./models/guo/monocular_model.py


echo "Preparing weight file ..."
cd ./models
python3 prepare_guo_weight.py
cd ../
echo "#### Finish installing ####"
