# Create a directory for Zsh plugins
mkdir -p "$HOME/.zsh"

# Clone the Pure theme
git clone https://github.com/sindresorhus/pure.git "$HOME/.zsh/pure"

# Clone the fast-syntax-highlighting plugin
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git "$HOME/.zsh/zsh-syntax-highlighting"

# Clone the zsh-autosuggestions plugin
git clone https://github.com/zsh-users/zsh-autosuggestions "$HOME/.zsh/zsh-autosuggestions"

# Clone the zsh-completions plugin
git clone https://github.com/zsh-users/zsh-completions "$HOME/.zsh/zsh-completions"

sudo apt update -y
sudo apt upgrade -y
sudo apt install zsh -y

cp zshrc ~/.zshrc
