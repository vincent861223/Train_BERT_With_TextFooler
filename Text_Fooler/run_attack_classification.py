import os

# for wordLSTM target
# command = 'python attack_classification.py --dataset_path data/yelp ' \
#           '--target_model wordLSTM --batch_size 128 ' \
#           '--target_model_path /scratch/jindi/adversary/BERT/results/yelp ' \
#           '--word_embeddings_path /data/medg/misc/jindi/nlp/embeddings/glove.6B/glove.6B.200d.txt ' \
#           '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /scratch/jindi/tf_cache'

# for BERT target
command = 'python attack_classification.py --dataset_path ../data/imdb_train ' \
          '--target_model bert ' \
          '--target_model_path ../model/pytorch_model.bin ' \
          '--max_seq_length 64 --batch_size 32 --output_dir results/imdb_result1 ' \
          '--counter_fitting_embeddings_path ../data/counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ../data/cos_sim_counter_fitting.npy ' \
          '--USE_cache_path None'

os.system(command)