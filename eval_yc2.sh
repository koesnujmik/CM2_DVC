############################################################################# yc2
# conda activate cm2
exp_name=03252250
config_path=cfgs/yc2_clip_cm2.yml
eval_folder=yc2_test
python eval.py --text_crossAttn_loc after --target_domain yc2 --bank_type yc2 --sim_match window_cos --ret_text token --down_proj deep --ret_vector nvec --eval_folder ${eval_folder} --exp_name ${exp_name} --eval_transformer_input_type queries --able_ret --sim_attention cls_token --soft_k 80 --window_size 50
# python eval.py --bank_type yc2 --text_crossAttn --text_crossAttn_loc after --sim_match window_cos --ret_text token --target_domain yc2 --down_proj deep --eval_folder ${eval_folder} --exp_name ${exp_name} --eval_transformer_input_type queries --able_ret --sim_attention cls_token 