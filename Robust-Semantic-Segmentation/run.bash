# uv run python main_px.py --config pspnet_ddcat_voc --attack_pixel 0.05 
# uv run python main_px.py --config pspnet_sat_voc --attack_pixel 0.05 

# uv run python main_px.py --config deeplabv3_ddcat_voc --attack_pixel 0.05
# uv run python main_px.py --config deeplabv3_sat_voc --attack_pixel 0.05

# uv run python main_sr.py --config pspnet_sat_voc --loss decision_change
# uv run python main_sr.py --config pspnet_ddcat_voc --loss decision_change

# uv run python main_sr.py --config deeplabv3_sat_voc --loss decision_change
# uv run python main_sr.py --config deeplabv3_ddcat_voc --loss decision_change


for model in deeplabv3_sat_city deeplabv3_ddcat_city pspnet_sat_city pspnet_ddcat_city; do
    for t in 0.1; do
        python pw_eval.py --config ${model} --max_query 1000 --npix 0.1 --num_images 100 --attack_mode scheduling --success_threshold $t
    done
done

for model in deeplabv3_sat_city deeplabv3_ddcat_city pspnet_sat_city pspnet_ddcat_city; do
    for t in 0.2 0.3; do
        python spaevo_eval.py --config ${model} --max_query 1000 --num_images 100 --n_pix 1960 --success_threshold $t
    done
done