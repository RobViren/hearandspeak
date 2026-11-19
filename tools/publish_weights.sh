#!/bin/bash
# Uploads what this crate hosts to rodbiren/kokoro_parakeet_aio_models: the int8 parakeet-ultra and the lexicon
# baked from ipa_everywhere's current lexicon. Rerun after lexicon edits.
# tools/publish_weights.sh [lexicon_only]
set -e
cd "$(dirname "$0")/.."
HF=${HF:-$HOME/Documents/cotg/server/.venv/bin/hf}
export HF_HUB_OFFLINE=0
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/hearandspeak/g2p"
python3 -I tools/bake_lexicon.py "$HOME/Documents/kokoallovero/weights/g2p/lexicon.bin" "$T/hearandspeak/g2p/lexicon.bin"
if [ "$1" != lexicon_only ]; then
  mkdir -p "$T/parakeet-ultra"
  cargo run -q --release --example convert_parakeet -- "$HOME/Documents/parakeetogo/model/model.safetensors" "$T/parakeet-ultra/model.safetensors"
  cp "$HOME/Documents/parakeetogo/model/tokenizer.json" "$T/parakeet-ultra/"
fi
"$HF" upload rodbiren/kokoro_parakeet_aio_models "$T" . --commit-message "hearandspeak: int8 parakeet-ultra, lexicon from ipa_everywhere"
