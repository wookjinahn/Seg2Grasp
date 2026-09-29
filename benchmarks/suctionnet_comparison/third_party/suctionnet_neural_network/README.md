# Vendored: SuctionNet learned model code (DeepLabV3+ / ConvNet)

`DeepLabV3Plus/` and `ConvNet/` are vendored from
[graspnet/suctionnet-baseline](https://github.com/graspnet/suctionnet-baseline)
(`neural_network/`) — the supervised RGB-D suction-heatmap models used as the
"learned" comparison arm (see `../../README.md` and `../../../../.ai/DECISIONS.md`,
2026-09-28 entry).

**Local patch (6 files):** `torchvision.models.utils.load_state_dict_from_url` was
removed in modern torchvision; each of these backbones imports it from
`torch.hub` instead (identical function, just relocated upstream):
`ConvNet/backbone/resnet.py`, `ConvNet/backbone/resnetRGBD.py`,
`DeepLabV3Plus/network/backbone/mobilenetv2.py`,
`DeepLabV3Plus/network/backbone/resnet.py`,
`DeepLabV3Plus/network/backbone/resnetDepth.py`,
`DeepLabV3Plus/network/backbone/resnetRGBD.py`.

No other changes. The adapter script that drives this code
(`../../suctionnet_nn_infer.py`) is a portable rewrite of
`neural_network/inference.py` from the same upstream repo — see its docstring.

See the upstream repo for license terms.
