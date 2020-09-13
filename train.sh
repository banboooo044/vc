python3 main.py \
-c config.yaml \
-d /home/data/mel_data \
-train_set train_128 \
-train_index_file train_samples_128.json \
--use_eval_set \
-eval_set in_test \
-eval_index_file in_test_samples_128.json \
--use_test_set \
-test_set out_test \
-test_index_file out_test_samples_128.json \
-logdir /home/vc/log \
-store_model_path /home/vc/model \
-tag exp3 \
-iters 500000 \
-summary_step 100