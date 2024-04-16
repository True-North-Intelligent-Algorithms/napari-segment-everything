# napari-segment-everything

[![License BSD-3](https://img.shields.io/pypi/l/napari-segment-everything.svg?color=green)](https://github.com/True-North-Intelligent-Algorithms/napari-segment-everything/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-segment-everything.svg?color=green)](https://pypi.org/project/napari-segment-everything)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-segment-everything.svg?color=green)](https://python.org)
[![tests](https://github.com/True-North-Intelligent-Algorithms/napari-segment-everything/workflows/tests/badge.svg)](https://github.com/True-North-Intelligent-Algorithms/napari-segment-everything/actions)
[![codecov](https://codecov.io/gh/True-North-Intelligent-Algorithms/napari-segment-everything/branch/main/graph/badge.svg)](https://codecov.io/gh/True-North-Intelligent-Algorithms/napari-segment-everything)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-segment-everything)](https://napari-hub.org/plugins/napari-segment-everything)

A Napari SAM plugin to segment everything in your image (not just some things)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

https://github.com/True-North-Intelligent-Algorithms/napari-segment-everything/assets/4366342/1f451e4a-bf66-4b77-a91d-4fa283270160

## Installation

You can install `napari-segment-everything` via [pip]:

    pip install napari-segment-everything

## Instructions

### 0. To use the mobileSAMv2 model, some additional setup is needed!
Model weights need to be first downloaded from here: https://github.com/ChaoningZhang/MobileSAM/tree/master?tab=readme-ov-file#getting-started-mobilesamv2
- Unzip the models
- Put the "ObjectAwareModel.pt" in 'napari-segment-everything/src/napari\_segment\_everything\minimalDetection\object\_detection\weight' directory.
- Put the other pytorch models in 'napari-segment-everything/src/napari\_segment\_everything\minimalDetection\weight' directory. 

### 1. Generate 3D labels

In the first step adjust SAM settings and generate a 3D representation of your labels.  The 3D view is needed to represent overlapping labels (labels that overlap in xy can be represented at different z).  After tweaking settings press 'Generate 3D labels'.  Be patient.  SAM with permissive settings can potentially find thousands of labels in a complicated image.  At least 6G of GPU memory is recommended to run SAM and to render to 3D label map (which can be large). 

### 2. Filter 3D labels

In the next step select a stat (solidity, hue, IOE, stability and other stats are available) then use the sliders and number boxes to filter out labels that do not represent structure of interest.  If you double click on a label a popup will appear containing the stats for that label.  Inpect stats for labels you want to keep, and labels you want to eliminate to help determine the filter settings. 

### 3. Generate 2D labels

In this step the 3D labels are projected to a 2D label map, use the dropdown to choose between projecting big labels in front or small labels in front.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-segment-everything" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
