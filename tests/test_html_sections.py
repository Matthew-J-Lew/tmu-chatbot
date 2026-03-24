import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ingestion.html_sections import extract_html_document


SAMPLE_HTML = """
<html>
  <head><title>Programs - Faculty of Arts - Toronto Metropolitan University (TMU)</title></head>
  <body>
    <main>
      <div class="resText"><div class="res-text"><h1>Undergraduate programs</h1><p>Study what you love.</p></div></div>
      <h2>Explore program options</h2>
      <p>13 undergraduate programs. 10 departments. Explore your options and choose the path that's right for you.</p>
      <div class="panel-group accordion background-opaque" id="abc" role="tablist">
        <div class="panel panel-default">
          <div class="panel-heading container-clipboard"><h3 class="panel-title"><a>Arts and Contemporary Studies - BA (Hons)</a></h3></div>
          <div class="panel-collapse collapse in"><div class="panel-body"><div class="resTwoColEven section"><div class="row">
            <div class="col-lg-6 a-b"><div class="c1 stackparsys">
              <div class="resText parbase section"><div class="res-text">
                <p>Shape your future by learning from the thinkers, artists and activists whose ideas have shaped our world.</p>
                <h4>Your program</h4>
                <ul><li>Full time: 4 years</li></ul>
                <h4>Your future career</h4>
                <p>Arts Administrator, Content Specialist</p>
              </div></div>
              <div class="resButtons section"><div class="res-buttons"><span>Explore program requirements for Arts and Contemporary Studies</span></div></div>
            </div></div>
            <div class="col-lg-6 a-b"><div class="c2 stackparsys">
              <div class="resImageText section"><img alt="Three students sit at a white table in a modern study space, deep in a lively discussion."></div>
              <div class="resText parbase section"><div class="res-text"><blockquote>My experience in this program has been so fulfilling.</blockquote><p>— Student</p></div></div>
            </div></div>
          </div></div></div></div>
        </div>
        <div class="panel panel-default">
          <div class="panel-heading container-clipboard"><h3 class="panel-title"><a>Criminology - BA (Hons)</a></h3></div>
          <div class="panel-collapse collapse"><div class="panel-body"><div class="resTwoColEven section"><div class="row">
            <div class="col-lg-6 a-b"><div class="c1 stackparsys">
              <div class="resText parbase section"><div class="res-text">
                <p>Are you passionate about social and political issues and how they relate to crime?</p>
                <h4>Your program</h4>
                <ul><li>Full time: 4 years</li><li>Full-time co-op: 5 years</li></ul>
                <h4>Your future career</h4>
                <p>Criminal Justice Program Developer, Court Reporter</p>
              </div></div>
            </div></div>
            <div class="col-lg-6 a-b"><div class="c2 stackparsys">
              <div class="resInfographic section"><div class="textContainer"><p><span class="resInfographicBold">12-16 </span></p><p>months of paid work experience available through co-op</p></div></div>
            </div></div>
          </div></div></div></div>
        </div>
      </div>
      <h2>Customize your degree</h2>
      <p>Graduate with a customized degree that’s as unique as you are.</p>
    </main>
  </body>
</html>
"""


def test_extract_html_document_handles_accordion_panels_as_hard_sections():
    doc = extract_html_document(SAMPLE_HTML, "https://www.torontomu.ca/arts/undergraduate/programs/")

    sections = [b.section for b in doc.blocks]
    assert "Explore program options" in sections
    assert "Arts and Contemporary Studies - BA (Hons)" in sections
    assert "Criminology - BA (Hons)" in sections
    assert "Customize your degree" in sections

    summary = next(b for b in doc.blocks if b.kind == "accordion_summary")
    assert summary.section == "Explore program options"
    assert "13 undergraduate programs" in summary.text
    assert "1. Arts and Contemporary Studies - BA (Hons)" in summary.text
    assert "2. Criminology - BA (Hons)" in summary.text
    assert "Official program list" not in summary.text

    arts = next(b for b in doc.blocks if b.section == "Arts and Contemporary Studies - BA (Hons)" and b.kind == "accordion_panel")
    assert "Shape your future" in arts.text
    assert "Content Specialist" in arts.text
    assert "Explore program requirements" not in arts.text
    assert "Three students sit at a white table" not in arts.text
    assert "Criminology - BA (Hons)" not in arts.text

    crim = next(b for b in doc.blocks if b.section == "Criminology - BA (Hons)" and b.kind == "accordion_panel")
    assert "Are you passionate" in crim.text
    assert "Court Reporter" in crim.text
    assert "12-16" in crim.text
    assert "months of paid work experience" in crim.text

    assert all(s not in {"12-16", "$65K", "1,600+"} for s in sections)
