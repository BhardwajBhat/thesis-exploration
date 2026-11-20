# reactor

Thesis code written on reactor (RTX 3090 Ti + KV260 over JTAG). Lives on branch `reactor`.

- Sources, small configs and small results only. Build outputs, bitstreams, weights and large logs are ignored (see `.gitignore`).
- Existing KV260 decoder work stays in `~/qwen-quant` (BhardwajBhat/qwen-quant); it is not copied here.
- KV260: bare metal, U-Boot only, temporary JTAG boots. A JTAG load resets the board; check `pgrep -a hw_server; fuser -v /dev/ttyUSB*` and ask the user before using it (it may be on loan).
