# Install Guide for mgf‑rust‑parser on Berzelius

## Add the following to your bashrc (or equivalent):

### 1. Install Rust (if not already installed):
curl https://sh.rustup.rs -sSf | sh
. "$HOME/.cargo/env"

### 2. Load the GCC build environment module:
module load buildenv-gcccuda/12.1.1-gcc12.3.0

### 3. Override the default linker so Rust uses gcc:
Add this line to your bashrc:
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/software/sse/manual/GCC/12.3.0/bin/gcc

### 4. Install protoc (Protocol Buffers compiler):
- Download and unzip protoc:
  cd ~/downloads
  wget https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-linux-x86_64.zip
  unzip protoc-21.12-linux-x86_64.zip -d protoc

- Set the PROTOC environment variable:
Add this line to your bashrc:
export PROTOC=/path/to/protoc/bin/protoc

Then, after reloading your bashrc, verify with:
echo $PROTOC
$PROTOC --version

### 5. Clone and build mgf‑rust‑parser:
git clone git@github.com:statisticalbiotechnology/mgf-rust-parser.git
cd mgf-rust-parser
cargo clean && cargo build --release
