#!/usr/bin/env bash

# Command line for experiment with job_id=imp_1_clients_per_round_1_job_id_18798
python run.py --rounds "100000" --client-sampling-type "important-sampling-with-replacement" --num-clients-per-round "1" --global-lr "0.1" --global-optimiser "sgd" --global-weight-decay "0.0" --number-of-local-iters "1" --run-local-steps --batch-size "500" --local-lr "1.0" --local-optimiser "sgd" --local-weight-decay "0.0" --dataset "australian" --loss "logistic" --model "logistic" --metric "top_1_acc" --global-regulizer "noncvx_robust_linear_regression" --global-regulizer-alpha "0.001" --checkpoint-dir "../check_points" --data-path "../data/" --gpu "-1" --log-gpu-usage --num-workers-train "0" --num-workers-test "0" --deterministic --manual-init-seed "11749" --manual-runtime-seed "15099" --group-name "internal_sgd:sgd-multi,sgd_multi_case_2,tau:1%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --comment "" --hostname "desktop-n968j9a" --eval-every "100" --eval-async-threads "1" --save-async-threads "1" --threadpool-for-local-opt "0" --run-id "imp_1_clients_per_round_1_job_id_18798" --algorithm "page" --algorithm-options "internal_sgd:sgd-multi,sgd_multi_case_2,tau:1%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --logfile "../logs/imp_1_clients_per_round_1_job_id_18798.txt" --client-compressor "ident" --extra-track "full_gradient_norm_train,full_objective_value_train" --allow-use-nv-tensorcores --initialize-shifts-policy "zero" --wandb-key "" --loglevel "debug" --logfilter ".*" --out "imp_1_clients_per_round_1_job_id_18798.bin" --out "imp_1_clients_per_round_1_job_id_18798.bin"

# Command line for experiment with job_id=imp_2_clients_per_round_1_job_id_29162
python run.py --rounds "100000" --client-sampling-type "important-sampling-with-replacement" --num-clients-per-round "1" --global-lr "0.1" --global-optimiser "sgd" --global-weight-decay "0.0" --number-of-local-iters "1" --run-local-steps --batch-size "500" --local-lr "1.0" --local-optimiser "sgd" --local-weight-decay "0.0" --dataset "australian" --loss "logistic" --model "logistic" --metric "top_1_acc" --global-regulizer "noncvx_robust_linear_regression" --global-regulizer-alpha "0.001" --checkpoint-dir "../check_points" --data-path "../data/" --gpu "-1" --log-gpu-usage --num-workers-train "0" --num-workers-test "0" --deterministic --manual-init-seed "4973" --manual-runtime-seed "29966" --group-name "internal_sgd:sgd-multi,sgd_multi_case_2,tau:10%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --comment "" --hostname "desktop-n968j9a" --eval-every "100" --eval-async-threads "1" --save-async-threads "1" --threadpool-for-local-opt "0" --run-id "imp_2_clients_per_round_1_job_id_29162" --algorithm "page" --algorithm-options "internal_sgd:sgd-multi,sgd_multi_case_2,tau:10%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --logfile "../logs/imp_2_clients_per_round_1_job_id_29162.txt" --client-compressor "ident" --extra-track "full_gradient_norm_train,full_objective_value_train" --allow-use-nv-tensorcores --initialize-shifts-policy "zero" --wandb-key "" --loglevel "debug" --logfilter ".*" --out "imp_2_clients_per_round_1_job_id_29162.bin" --out "imp_2_clients_per_round_1_job_id_29162.bin"

# Command line for experiment with job_id=imp_3_clients_per_round_1_job_id_27919
python run.py --rounds "100000" --client-sampling-type "important-sampling-with-replacement" --num-clients-per-round "1" --global-lr "0.1" --global-optimiser "sgd" --global-weight-decay "0.0" --number-of-local-iters "1" --run-local-steps --batch-size "500" --local-lr "1.0" --local-optimiser "sgd" --local-weight-decay "0.0" --dataset "australian" --loss "logistic" --model "logistic" --metric "top_1_acc" --global-regulizer "noncvx_robust_linear_regression" --global-regulizer-alpha "0.001" --checkpoint-dir "../check_points" --data-path "../data/" --gpu "-1" --log-gpu-usage --num-workers-train "0" --num-workers-test "0" --deterministic --manual-init-seed "29644" --manual-runtime-seed "36" --group-name "internal_sgd:sgd-multi,sgd_multi_case_2,tau:30%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --comment "" --hostname "desktop-n968j9a" --eval-every "100" --eval-async-threads "1" --save-async-threads "1" --threadpool-for-local-opt "0" --run-id "imp_3_clients_per_round_1_job_id_27919" --algorithm "page" --algorithm-options "internal_sgd:sgd-multi,sgd_multi_case_2,tau:30%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --logfile "../logs/imp_3_clients_per_round_1_job_id_27919.txt" --client-compressor "ident" --extra-track "full_gradient_norm_train,full_objective_value_train" --allow-use-nv-tensorcores --initialize-shifts-policy "zero" --wandb-key "" --loglevel "debug" --logfilter ".*" --out "imp_3_clients_per_round_1_job_id_27919.bin" --out "imp_3_clients_per_round_1_job_id_27919.bin"

# Command line for experiment with job_id=imp_4_clients_per_round_1_job_id_20995
python run.py --rounds "100000" --client-sampling-type "important-sampling-with-replacement" --num-clients-per-round "1" --global-lr "0.1" --global-optimiser "sgd" --global-weight-decay "0.0" --number-of-local-iters "1" --run-local-steps --batch-size "500" --local-lr "1.0" --local-optimiser "sgd" --local-weight-decay "0.0" --dataset "australian" --loss "logistic" --model "logistic" --metric "top_1_acc" --global-regulizer "noncvx_robust_linear_regression" --global-regulizer-alpha "0.001" --checkpoint-dir "../check_points" --data-path "../data/" --gpu "-1" --log-gpu-usage --num-workers-train "0" --num-workers-test "0" --deterministic --manual-init-seed "32372" --manual-runtime-seed "8349" --group-name "internal_sgd:sgd-multi,sgd_multi_case_2,tau:50%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --comment "" --hostname "desktop-n968j9a" --eval-every "100" --eval-async-threads "1" --save-async-threads "1" --threadpool-for-local-opt "0" --run-id "imp_4_clients_per_round_1_job_id_20995" --algorithm "page" --algorithm-options "internal_sgd:sgd-multi,sgd_multi_case_2,tau:50%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --logfile "../logs/imp_4_clients_per_round_1_job_id_20995.txt" --client-compressor "ident" --extra-track "full_gradient_norm_train,full_objective_value_train" --allow-use-nv-tensorcores --initialize-shifts-policy "zero" --wandb-key "" --loglevel "debug" --logfilter ".*" --out "imp_4_clients_per_round_1_job_id_20995.bin" --out "imp_4_clients_per_round_1_job_id_20995.bin"

# Command line for experiment with job_id=imp_5_clients_per_round_1_job_id_8905
python run.py --rounds "100000" --client-sampling-type "important-sampling-with-replacement" --num-clients-per-round "1" --global-lr "0.1" --global-optimiser "sgd" --global-weight-decay "0.0" --number-of-local-iters "1" --run-local-steps --batch-size "500" --local-lr "1.0" --local-optimiser "sgd" --local-weight-decay "0.0" --dataset "australian" --loss "logistic" --model "logistic" --metric "top_1_acc" --global-regulizer "noncvx_robust_linear_regression" --global-regulizer-alpha "0.001" --checkpoint-dir "../check_points" --data-path "../data/" --gpu "-1" --log-gpu-usage --num-workers-train "0" --num-workers-test "0" --deterministic --manual-init-seed "21772" --manual-runtime-seed "5430" --group-name "internal_sgd:sgd-multi,sgd_multi_case_2,tau:80%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --comment "" --hostname "desktop-n968j9a" --eval-every "100" --eval-async-threads "1" --save-async-threads "1" --threadpool-for-local-opt "0" --run-id "imp_5_clients_per_round_1_job_id_8905" --algorithm "page" --algorithm-options "internal_sgd:sgd-multi,sgd_multi_case_2,tau:80%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --logfile "../logs/imp_5_clients_per_round_1_job_id_8905.txt" --client-compressor "ident" --extra-track "full_gradient_norm_train,full_objective_value_train" --allow-use-nv-tensorcores --initialize-shifts-policy "zero" --wandb-key "" --loglevel "debug" --logfilter ".*" --out "imp_5_clients_per_round_1_job_id_8905.bin" --out "imp_5_clients_per_round_1_job_id_8905.bin"

# Command line for experiment with job_id=uni_1_clients_per_round_1_job_id_3414
python run.py --rounds "100000" --client-sampling-type "uniform-sampling-with-replacement" --num-clients-per-round "1" --global-lr "0.1" --global-optimiser "sgd" --global-weight-decay "0.0" --number-of-local-iters "1" --run-local-steps --batch-size "500" --local-lr "1.0" --local-optimiser "sgd" --local-weight-decay "0.0" --dataset "australian" --loss "logistic" --model "logistic" --metric "top_1_acc" --global-regulizer "noncvx_robust_linear_regression" --global-regulizer-alpha "0.001" --checkpoint-dir "../check_points" --data-path "../data/" --gpu "-1" --log-gpu-usage --num-workers-train "0" --num-workers-test "0" --deterministic --manual-init-seed "6814" --manual-runtime-seed "3332" --group-name "internal_sgd:sgd-multi,sgd_multi_case_1,tau:1%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --comment "" --hostname "desktop-n968j9a" --eval-every "100" --eval-async-threads "1" --save-async-threads "1" --threadpool-for-local-opt "0" --run-id "uni_1_clients_per_round_1_job_id_3414" --algorithm "page" --algorithm-options "internal_sgd:sgd-multi,sgd_multi_case_1,tau:1%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --logfile "../logs/uni_1_clients_per_round_1_job_id_3414.txt" --client-compressor "ident" --extra-track "full_gradient_norm_train,full_objective_value_train" --allow-use-nv-tensorcores --initialize-shifts-policy "zero" --wandb-key "" --loglevel "debug" --logfilter ".*" --out "uni_1_clients_per_round_1_job_id_3414.bin" --out "uni_1_clients_per_round_1_job_id_3414.bin"

# Command line for experiment with job_id=uni_2_clients_per_round_1_job_id_877
python run.py --rounds "100000" --client-sampling-type "uniform-sampling-with-replacement" --num-clients-per-round "1" --global-lr "0.1" --global-optimiser "sgd" --global-weight-decay "0.0" --number-of-local-iters "1" --run-local-steps --batch-size "500" --local-lr "1.0" --local-optimiser "sgd" --local-weight-decay "0.0" --dataset "australian" --loss "logistic" --model "logistic" --metric "top_1_acc" --global-regulizer "noncvx_robust_linear_regression" --global-regulizer-alpha "0.001" --checkpoint-dir "../check_points" --data-path "../data/" --gpu "-1" --log-gpu-usage --num-workers-train "0" --num-workers-test "0" --deterministic --manual-init-seed "16640" --manual-runtime-seed "29494" --group-name "internal_sgd:sgd-multi,sgd_multi_case_1,tau:10%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --comment "" --hostname "desktop-n968j9a" --eval-every "100" --eval-async-threads "1" --save-async-threads "1" --threadpool-for-local-opt "0" --run-id "uni_2_clients_per_round_1_job_id_877" --algorithm "page" --algorithm-options "internal_sgd:sgd-multi,sgd_multi_case_1,tau:10%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --logfile "../logs/uni_2_clients_per_round_1_job_id_877.txt" --client-compressor "ident" --extra-track "full_gradient_norm_train,full_objective_value_train" --allow-use-nv-tensorcores --initialize-shifts-policy "zero" --wandb-key "" --loglevel "debug" --logfilter ".*" --out "uni_2_clients_per_round_1_job_id_877.bin" --out "uni_2_clients_per_round_1_job_id_877.bin"

# Command line for experiment with job_id=uni_3_clients_per_round_1_job_id_8243
python run.py --rounds "100000" --client-sampling-type "uniform-sampling-with-replacement" --num-clients-per-round "1" --global-lr "0.1" --global-optimiser "sgd" --global-weight-decay "0.0" --number-of-local-iters "1" --run-local-steps --batch-size "500" --local-lr "1.0" --local-optimiser "sgd" --local-weight-decay "0.0" --dataset "australian" --loss "logistic" --model "logistic" --metric "top_1_acc" --global-regulizer "noncvx_robust_linear_regression" --global-regulizer-alpha "0.001" --checkpoint-dir "../check_points" --data-path "../data/" --gpu "-1" --log-gpu-usage --num-workers-train "0" --num-workers-test "0" --deterministic --manual-init-seed "5357" --manual-runtime-seed "31614" --group-name "internal_sgd:sgd-multi,sgd_multi_case_1,tau:30%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --comment "" --hostname "desktop-n968j9a" --eval-every "100" --eval-async-threads "1" --save-async-threads "1" --threadpool-for-local-opt "0" --run-id "uni_3_clients_per_round_1_job_id_8243" --algorithm "page" --algorithm-options "internal_sgd:sgd-multi,sgd_multi_case_1,tau:30%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --logfile "../logs/uni_3_clients_per_round_1_job_id_8243.txt" --client-compressor "ident" --extra-track "full_gradient_norm_train,full_objective_value_train" --allow-use-nv-tensorcores --initialize-shifts-policy "zero" --wandb-key "" --loglevel "debug" --logfilter ".*" --out "uni_3_clients_per_round_1_job_id_8243.bin" --out "uni_3_clients_per_round_1_job_id_8243.bin"

# Command line for experiment with job_id=uni_4_clients_per_round_1_job_id_12087
python run.py --rounds "100000" --client-sampling-type "uniform-sampling-with-replacement" --num-clients-per-round "1" --global-lr "0.1" --global-optimiser "sgd" --global-weight-decay "0.0" --number-of-local-iters "1" --run-local-steps --batch-size "500" --local-lr "1.0" --local-optimiser "sgd" --local-weight-decay "0.0" --dataset "australian" --loss "logistic" --model "logistic" --metric "top_1_acc" --global-regulizer "noncvx_robust_linear_regression" --global-regulizer-alpha "0.001" --checkpoint-dir "../check_points" --data-path "../data/" --gpu "-1" --log-gpu-usage --num-workers-train "0" --num-workers-test "0" --deterministic --manual-init-seed "168" --manual-runtime-seed "7612" --group-name "internal_sgd:sgd-multi,sgd_multi_case_1,tau:50%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --comment "" --hostname "desktop-n968j9a" --eval-every "100" --eval-async-threads "1" --save-async-threads "1" --threadpool-for-local-opt "0" --run-id "uni_4_clients_per_round_1_job_id_12087" --algorithm "page" --algorithm-options "internal_sgd:sgd-multi,sgd_multi_case_1,tau:50%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --logfile "../logs/uni_4_clients_per_round_1_job_id_12087.txt" --client-compressor "ident" --extra-track "full_gradient_norm_train,full_objective_value_train" --allow-use-nv-tensorcores --initialize-shifts-policy "zero" --wandb-key "" --loglevel "debug" --logfilter ".*" --out "uni_4_clients_per_round_1_job_id_12087.bin" --out "uni_4_clients_per_round_1_job_id_12087.bin"

# Command line for experiment with job_id=uni_5_clients_per_round_1_job_id_21498
python run.py --rounds "100000" --client-sampling-type "uniform-sampling-with-replacement" --num-clients-per-round "1" --global-lr "0.1" --global-optimiser "sgd" --global-weight-decay "0.0" --number-of-local-iters "1" --run-local-steps --batch-size "500" --local-lr "1.0" --local-optimiser "sgd" --local-weight-decay "0.0" --dataset "australian" --loss "logistic" --model "logistic" --metric "top_1_acc" --global-regulizer "noncvx_robust_linear_regression" --global-regulizer-alpha "0.001" --checkpoint-dir "../check_points" --data-path "../data/" --gpu "-1" --log-gpu-usage --num-workers-train "0" --num-workers-test "0" --deterministic --manual-init-seed "21983" --manual-runtime-seed "13674" --group-name "internal_sgd:sgd-multi,sgd_multi_case_1,tau:80%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --comment "" --hostname "desktop-n968j9a" --eval-every "100" --eval-async-threads "1" --save-async-threads "1" --threadpool-for-local-opt "0" --run-id "uni_5_clients_per_round_1_job_id_21498" --algorithm "page" --algorithm-options "internal_sgd:sgd-multi,sgd_multi_case_1,tau:80%,use_optimal_p,th_stepsize_noncvx,page_ab_synthetic,logregression,x0_norm:0.0" --logfile "../logs/uni_5_clients_per_round_1_job_id_21498.txt" --client-compressor "ident" --extra-track "full_gradient_norm_train,full_objective_value_train" --allow-use-nv-tensorcores --initialize-shifts-policy "zero" --wandb-key "" --loglevel "debug" --logfilter ".*" --out "uni_5_clients_per_round_1_job_id_21498.bin" --out "uni_5_clients_per_round_1_job_id_21498.bin"

