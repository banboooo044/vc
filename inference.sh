python3 inference.py \
-attr /Users/banboooo044/Documents/speaker_class/data/attr.pkl \
-c /Users/banboooo044/Documents/vc/config.yaml \
-m /Users/banboooo044/Documents/vc/model/exp2.ckpt \
--gpu_model \
-s /Users/banboooo044/Documents/vc_colab/VCTK-Corpus/wav48/p225/p225_001.wav \
-t /Users/banboooo044/Documents/vc_colab/VCTK-Corpus/wav48/p226/p226_001.wav \
-o /Users/banboooo044/Documents/vc/output_wav/225_226_001.wav