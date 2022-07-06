declare -a datasets=("residential_building" "enb" "scpf")

for name in ${datasets[@]}; do

  echo $name
#
  python conformal_real_data.py  \
          --base_path ../input/$name \
          --data_path $name.csv \
		  --data_name $name \
          > ../input/$name/log_file.log
done


