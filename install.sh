%cd /content/

%mkdir model_data
!wget -O /content/model_data/vocabulary_captioning_thresh5.txt https://dl.fbaipublicfiles.com/pythia/data/vocabulary_captioning_thresh5.txt
!wget -O /content/model_data/detectron_model.pth  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth 
!wget -O /content/model_data/butd.pth https://dl.fbaipublicfiles.com/pythia/pretrained_models/coco_captions/butd.pth
!wget -O /content/model_data/butd.yaml https://dl.fbaipublicfiles.com/pythia/pretrained_models/coco_captions/butd.yml
!wget -O /content/model_data/detectron_model.yaml https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
!wget -O /content/model_data/detectron_weights.tar.gz https://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
!tar xf /content/model_data/detectron_weights.tar.gz

