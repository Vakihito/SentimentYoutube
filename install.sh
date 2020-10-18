if not exist "/content/" mkdir "/content/"

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

python3
import sys
sys.path.append('/content/pythia')
exit()

# Install maskrcnn-benchmark to extract detectron features
cd /content
git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git
cd /content/vqa-maskrcnn-benchmark
# Compile custom layers and build mask-rcnn backbone
python setup.py build
python setup.py develop

python3
import sys
sys.path.append('/content/vqa-maskrcnn-benchmark')


sys.path.append('/content')
exit()

!wget http://nlp.stanford.edu/data/glove.6B.zip

!mkdir -p /content/pythia/.vector_cache/
!mv glove.6B.zip /content/pythia/.vector_cache/

