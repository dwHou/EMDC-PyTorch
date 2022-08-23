# :<<!
for f in $(ls -h ./validation_input/)
do  
    name="EMDC-validation"
    metric="visual"
    mkdir -p ./results/${name}/${metric}
    echo ${f}
    mkdir ./results/${name}/${metric}/${f}
    python3 -u StoDense_plt.py --arch EMDC \
    --txt_path ./validation_input/${f}/data.list \
    --out_path ${name}/${metric}/${f} \
    --visualization 1 \
    --seemap 0 \
    --output_num 3 \
    --readme_path ${name}/${metric} \
    --ckp_path ../checkpoints/milestone.pth.tar \
    --gpu 2
done
# !