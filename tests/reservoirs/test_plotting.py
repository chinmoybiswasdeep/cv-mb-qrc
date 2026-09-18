import matplotlib.pyplot as plt

from cv_mb_qrc.reservoirs.plotting import OKABE_ITO, apply_publication_style, method_styles


def test_publication_style_is_colorblind_safe_and_encodes_method_type_twice():
    apply_publication_style()
    assert len(OKABE_ITO) == len(set(OKABE_ITO)) == 8
    styles = method_styles(["cv_A", "delay", "graphix"])
    assert styles["cv_A"]["linestyle"] != styles["delay"]["linestyle"]
    assert styles["cv_A"]["marker"] != styles["delay"]["marker"]
    assert plt.rcParams["savefig.dpi"] == 600
    assert plt.rcParams["pdf.fonttype"] == 42
