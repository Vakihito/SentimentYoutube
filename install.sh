cd /content/

mkdir model_data
wget -O /content/model_data/vocabulary_captioning_thresh5.txt https://dl.fbaipublicfiles.com/pythia/data/vocabulary_captioning_thresh5.txt
wget -O /content/model_data/detectron_model.pth  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth 
wget -O /content/model_data/butd.pth https://dl.fbaipublicfiles.com/pythia/pretrained_models/coco_captions/butd.pth
wget -O /content/model_data/butd.yaml https://dl.fbaipublicfiles.com/pythia/pretrained_models/coco_captions/butd.yml
wget -O /content/model_data/detectron_model.yaml https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
wget -O /content/model_data/detectron_weights.tar.gz https://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf /content/model_data/detectron_weights.tar.gz






wget -O /content/model_data/detectron_model.pth  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth 


cd /content/
rm -rf pythia
git clone https://github.com/facebookresearch/pythia.git pythia
cd /content/pythia
# Don't modify torch version
sed -i '/torch/d' requirements.txt
pip install -e .

export PYTHONPATH=/content/pythia
echo $PYTHONPATH


cd /content/
wget https://github.com/facebookresearch/mmf/archive/v0.3.zip
unzip v0.3.zip






cd /content/
rm -rf pythia
#!git clone https://github.com/facebookresearch/pythia.git pythia
cp -afxr mmf-0.3/pythia /content/pythia
cd /content/pythia


export PYTHONPATH=/content/pythia
echo $PYTHONPATH


# Install maskrcnn-benchmark to extract detectron features
cd /content
git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git
cd /content/vqa-maskrcnn-benchmark
# Compile custom layers and build mask-rcnn backbone
python setup.py build
python setup.py develop

export PYTHONPATH=/content/vqa-maskrcnn-benchmark
export PYTHONPATH=/content  
echo $PYTHONPATH

wget http://nlp.stanford.edu/data/glove.6B.zip

mkdir -p /content/pythia/.vector_cache/
mv glove.6B.zip /content/pythia/.vector_cache/


mv /content/SentimentYoutube/SentimentYoutube /content/vqa-maskrcnn-benchmark
mv /content/pythia /content/vqa-maskrcnn-benchmark

#download my trained modelss
cd /content
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZGIaBKmHa7C0qmxorgU53xnH-_twBNH2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZGIaBKmHa7C0qmxorgU53xnH-_twBNH2" -O trained_models.zip && rm -rf /tmp/cookies.txt
unzip trained_models.zip

pip3 install yacs
pip3 install git+https://github.com/nficano/pytube
pip3 install git+https://github.com/Vakihito/ktrain.git
pip3 install git+https://github.com/amaiya/eli5@tfkeras_0_10_1
pip3 install git+https://github.com/amaiya/stellargraph@no_tf_dep_082

wget https://github.com/gopinath-balu/computer_vision/raw/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel
wget https://raw.githubusercontent.com/gopinath-balu/computer_vision/master/CAFFE_DNN/deploy.prototxt.txt
#open the correct dir to use
pip3 install git+https://github.com/rmarcacini/sentence-transformers
pip3 install gdown
gdown https://drive.google.com/uc?id=1NV5t1YhyyOzMF5zAovfbSLdZZLvqrfZ_
unzip distiluse-base-multilingual-cased.zip -d language_model

gdown --id 1rzOmkZHWMpNLyA4XitU_i1sCN-NVLU52
unzip bert_lex_model-20201124T204602Z-001.zip
gdown --id 1eEHQ0NMJldPzvh9crE6C-3wV4DA1hIE5