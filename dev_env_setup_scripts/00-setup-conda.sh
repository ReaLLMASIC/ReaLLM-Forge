mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda init --all

env_name="reallmforge"
conda create --name reallmforge python=3.10
echo 'conda activate reallmforge' >> ~/.zshrc
echo 'conda activate reallmforge' >> ~/.bashrc

sudo apt install build-essential
sudo apt install python3-pip
