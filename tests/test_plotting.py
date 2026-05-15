import io
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

from prism.utils import mask_to_outline, plot_outline


def _circular_mask():
    ny = nx = 41
    y, x = np.indices((ny, nx))

    outer = (x - 20)**2 + (y - 20)**2 <= 8**2
    hole = (x - 20)**2 + (y - 20)**2 <= 2**2
    second = (x - 31)**2 + (y - 10)**2 <= 4**2

    return (outer & ~hole) | second


def test_plot_outline_returns_closed_paths_for_mask():
    mask = _circular_mask()

    fig, ax = plt.subplots()
    patches = plot_outline(mask, ax=ax, color='red', linewidth=2)

    assert len(patches) == 3
    for patch in patches:
        path = patch.get_path()
        assert path.codes[0] == Path.MOVETO
        assert path.codes[-1] == Path.CLOSEPOLY
        np.testing.assert_allclose(path.vertices[0], path.vertices[-1])

    plt.close(fig)


def test_plot_outline_svg_output_uses_closepath_for_segments():
    mask = _circular_mask()
    segments = mask_to_outline(mask)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_aspect('equal')
    plot_outline(segments, ax=ax, color='red', linewidth=2)

    buffer = io.StringIO()
    fig.savefig(buffer, format='svg', transparent=True)
    svg = buffer.getvalue()
    plt.close(fig)

    root = ET.fromstring(svg)
    outlines = []
    for elem in root.iter():
        if not elem.tag.endswith('path'):
            continue
        style = elem.attrib.get('style', '')
        if 'stroke: #ff0000' not in style:
            continue
        outlines.append(elem.attrib['d'])

    assert len(outlines) == 3
    assert all('z' in outline.lower() for outline in outlines)
