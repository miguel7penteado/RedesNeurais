# Redes Neurais


### Linux Debian
```
# repositório NVIDIA for Debian
# https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_network

wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-debian11-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-debian11-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-debian11-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo add-apt-repository contrib
sudo apt-get update

sudo apt install cuda-toolkit-12

```
