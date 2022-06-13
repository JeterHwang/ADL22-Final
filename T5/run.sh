pip install -r requirement.txt
python3 -m spacy en_core_web_sm

chmod 777 download.sh
./download.sh

python3 simulator.py --split test --device cuda:0