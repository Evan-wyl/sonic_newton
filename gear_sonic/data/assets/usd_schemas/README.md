# USD schema resources for standalone G1 import

These unmodified codeless USD schema resources are extracted from the official
`ovphysx==0.6.4.72118723` Linux x86_64 wheel selected by Isaac Lab v3.0.0's
`uv.lock`. They define asset metadata; no PhysX engine library is bundled or run.

Wheel SHA-256:
`8276824ef74bd79ae231804850a9dcaf218967c6dd92bc31f4d1bc1c1e0d4643`

The standalone Isaac Sim 6.2 URDF importer authors `PhysxJointAPI`. The separately
published `physx-usd-schemas==25.11.2.72696984` does not define that newer schema,
so it is insufficient for this importer. Register these resources before using
the importer. Keep the full schema resource set and accompanying licenses.

Source wheel:
https://files.pythonhosted.org/packages/5a/52/04625f93a57c6dbabc1d3326b6211d4983dfe3d13084fea1aefdba9edd4f/ovphysx-0.6.4.72118723-py3-none-manylinux_2_35_x86_64.whl
