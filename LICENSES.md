# Licenses

## Remotion License
Based on the [Remotion License FAQ](https://www.remotion.dev/license) (read on 2026-10-09):
- **Free for:** Individuals, organisations of up to 3 people, and non-profits.
- **Automated Rendering:** Falls under "Remotion for Automators". The pricing is $0.01 per render with a $100 monthly minimum.
- **Telemetry:** Mandatory for Automators starting in Remotion 5.0.
- **Patents:** Codec patents (such as H.264) are not covered by the license.

## Depth Anything V2 Small (depth parallax)
- **Model:** Depth-Anything-V2-Small, Apache-2.0 ([official repo](https://github.com/DepthAnything/Depth-Anything-V2), checked 2026-10-12). ONNX export from [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small) (Apache-2.0), pinned to revision `4472b736` in `modules/depth.py`.
- **Do not use** Depth-Anything-V2 Base/Large/Giant: they are CC-BY-NC-4.0 (non-commercial).
- Apache-2.0 requires shipping the license and any NOTICE with the weights.

## Rendering libraries
- `onnxruntime`, `three`, `@react-three/fiber`: MIT.
- `@remotion/three`: covered by the Remotion license above.
