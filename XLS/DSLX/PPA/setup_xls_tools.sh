#!/bin/bash

xls_version="v0.0.0-5199-gf59b8a886"
xls_colab_version="v0.0.0-5201-ge64c215d8"
yosys_version="0.38_93_g84116c9a3"
openroad_version="2.0_12381_g01bba3695"
rules_hdl_version="2eb050e80a5c42ac3ffdb7e70392d86a6896dfc7"


echo "📦 Downloading xls-${xls_version}"
curl -L "https://github.com/google/xls/releases/download/${xls_version}/xls-${xls_version}-linux-x64.tar.gz" | tar xzf - --strip-components=1




echo "🧪 Installing Python dependencies"
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir "https://github.com/proppy/xls/releases/download/${xls_colab_version}/xls_colab-0.0.0-py3-none-any.whl"



######################################### old ###############################################
# echo "🧪 Setting up colab integration"
# pip install --quiet --no-cache-dir --ignore-installed \
#     "https://github.com/proppy/xls/releases/download/${xls_colab_version}/xls_colab-0.0.0-py3-none-any.whl"
######################################### old ###############################################





##############################################################################################
echo "🛣️ Installing OpenROAD and related tools"
curl -L -O "https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh"
bash Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -b -p conda-env/
source conda-env/bin/activate
conda install -yq -c "litex-hub" openroad=${openroad_version} yosys=${yosys_version}
##############################################################################################



######################################### old ###############################################
# echo "🛣️ Installing openroad and yosys"
# curl -L -O "https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh"
# bash Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -b -p conda-env/

# export PATH=$PWD/conda-env/bin:$PATH


# python3 set_conda_prefix.py
# set -a
# source .env
# set +a
# echo "🧊 CONDA_PREFIX is set to: $CONDA_PREFIX"

# conda-env/bin/conda install -yq -c "litex-hub" openroad="${openroad_version}" yosys="${yosys_version}"
######################################### old ###############################################





# echo "🧊 Installing openroad dependencies"
# sudo apt-get install libfmt8
# sudo apt-get install libicu-dev

# # Idea Effect:
# # sudo ln -s /usr/lib/x86_64-linux-gnu/libicui18n.so.70 /usr/lib/x86_64-linux-gnu/libicui18n.so.58
# # echo "🧊 Soft Linking libicui18n.so.<N> to provide libicui18n.so.58 for OpenRoad"
# # chmod +x ./setup_softlink_libicui18n_58.sh
# # ./setup_softlink_libicui18n_58.sh

# # Redo Soft Linking:
# # sudo rm /usr/lib/x86_64-linux-gnu/libicui18n.so.58

# echo "🧊 Downloading and building libicui18n.so.58 for OpenRoad"
# chmod +x ./setup_install_libicui18n_so_58.sh
# ./setup_install_libicui18n_so_58.sh

# # For local test:
# # source /home/...<current_dir>/conda-env/bin/activate
# openroad --version
# exit 0


######################################### old ###############################################
# echo "🧰 Generating PDK metadata"
# curl -L "https://github.com/hdl/bazel_rules_hdl/archive/${rules_hdl_version}.tar.gz" | tar xzf - --strip-components=1

# echo "✅ Setup complete! You can now use the installed tools."




# chmod +x setup_install_dependencies.sh
# ./setup_install_dependencies.sh
######################################### old ###############################################


echo "🧰 Generating PDK metadata"
curl --show-error -L "https://github.com/hdl/bazel_rules_hdl/archive/${rules_hdl_version}.tar.gz" | tar xzf - --strip-components=1
curl -L -O "https://github.com/protocolbuffers/protobuf/releases/download/v24.3/protoc-24.3-linux-x86_64.zip"
unzip -q -o protoc-24.3-linux-x86_64.zip
python3 -m pip install protobuf


gsutil cp gs://proppy-eda/pdk_info_asap7.zip .
unzip -q -o pdk_info_asap7.zip


echo "🧊 Give Access to Everyone for asap7"
sudo chmod -R u+w org_theopenroadproject_asap7sc7p5t_28
sudo chmod -R u+w asap7/dependency_support/org_theopenroadproject_asap7_pdk_r1p7

mkdir -p org_theopenroadproject_asap7sc7p5t_28/{LEF,techlef_misc} asap7/dependency_support/org_theopenroadproject_asap7_pdk_r1p7/
cp asap7/asap7sc7p5t_28_R_1x_220121a.lef org_theopenroadproject_asap7sc7p5t_28/LEF/
cp asap7/asap7_tech_1x_201209.lef org_theopenroadproject_asap7sc7p5t_28/techlef_misc/
cp asap7/asap7_rvt_1x_SS.lib org_theopenroadproject_asap7sc7p5t_28/
cp asap7/tracks.tcl asap7/dependency_support/org_theopenroadproject_asap7_pdk_r1p7/
cp asap7/pdn_config.pdn asap7/dependency_support/org_theopenroadproject_asap7_pdk_r1p7/
cp asap7/rc_script.tcl asap7/dependency_support/org_theopenroadproject_asap7_pdk_r1p7/


bin/protoc --python_out=. pdk/proto/pdk_info.proto
ln -sf pdk/proto/pdk_info_pb2.py