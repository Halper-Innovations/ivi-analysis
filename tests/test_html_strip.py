"""Tests for app.util.html_strip."""


def test_strip_html_removes_tags():
    from app.util.html_strip import strip_html

    html = '<span style="color:red">Hello</span> <b>World</b>'
    assert strip_html(html) == "Hello World"


def test_strip_html_preserves_paragraphs():
    from app.util.html_strip import strip_html

    html = "<p>First paragraph.</p><p>Second paragraph.</p>"
    result = strip_html(html)
    assert "First paragraph." in result
    assert "Second paragraph." in result
    assert "\n" in result  # paragraphs separated


def test_strip_html_decodes_entities():
    from app.util.html_strip import strip_html

    html = "Revenue &amp; Earnings &gt; $1B"
    assert strip_html(html) == "Revenue & Earnings > $1B"


def test_strip_html_collapses_whitespace():
    from app.util.html_strip import strip_html

    html = "Too    many     spaces\n\n\n\n\nToo many newlines"
    result = strip_html(html)
    assert "  " not in result
    assert "\n\n\n" not in result


def test_strip_html_empty_input():
    from app.util.html_strip import strip_html

    assert strip_html("") == ""
    assert strip_html(None) is None


def test_strip_html_real_edgar_snippet():
    from app.util.html_strip import strip_html

    edgar_html = (
        '<div style="margin-bottom:6pt;margin-top:9pt;text-align:justify">'
        '<span style="color:#76b900;font-family:\'NVIDIA Sans\',sans-serif;'
        'font-size:9pt;font-weight:700;line-height:120%">Our Company</span>'
        '</div><div><span>NVIDIA pioneered accelerated computing.</span></div>'
    )
    result = strip_html(edgar_html)
    assert "<span" not in result
    assert "<div" not in result
    assert "Our Company" in result
    assert "NVIDIA pioneered accelerated computing." in result
