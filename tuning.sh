python3 tuning.py \
-c config2.yaml \
-d /home/data/mel_data \
-train_set train_128 \
-train_index_file train_samples_128.json \
--use_eval_set \
-eval_set in_test \
-eval_index_file in_test_samples_128.json \
-logdir /home/vc/log \
-store_model_path /home/vc/model \
-tag exp_tuning \
-iters 50000 \
-summary_step 100