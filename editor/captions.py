"""Captions + sound cues on a rendered 9:16 short with pycaps (github.com/francozanardi/pycaps, MIT):

  python editor/captions.py <in.mp4> <out.mp4> <captions.json> <y 0..1> [#active-word colour]

captions.json = whisper_json {"segments": [{"start", "end", "words": [{"word", "start", "end"}]}]} in OUTPUT seconds
(renderer.caption_doc: Whisper on the cut audio, so no drift). y = where the caption block's centre sits (fraction of
the height: renderer.CAPTION_Y per layout). Style: editor/pycaps_style (bold uppercase Komika Axis, active word
zoom-in + yellow, hype words green / red, a soft pop cue on hype words). Renders without a browser
(PictexSubtitleRenderer) and never calls an AI service (no "ai" tagger rules, no emoji effects).
renderer.burn_captions runs this as its own process: pycaps / movielite print progress bars and may spawn workers.
"""
import json
import sys
from pathlib import Path

STYLE = Path(__file__).resolve().parent / "pycaps_style" / "pycaps.template.json"


def main(argv):
    import pycaps
    from pycaps import JsonConfigLoader, PictexSubtitleRenderer
    from pycaps.layout import SubtitleLayoutOptions

    src, out, doc, y = argv[0], argv[1], json.loads(Path(argv[2]).read_text(encoding="utf-8")), float(argv[3])
    b = JsonConfigLoader(str(STYLE)).load(False)
    b.with_input_video(src).with_output_video(out).with_transcription(doc).should_save_subtitle_data(False)
    # pycaps' "center" puts the block's top at (H - h) * (offset + 0.5): offset y - 0.5 centres it at ~y
    layout = json.loads(STYLE.read_text(encoding="utf-8"))["layout"]
    b.with_layout_options(SubtitleLayoutOptions(**dict(layout, vertical_align={"align": "center", "offset": y - 0.5})))
    b.with_resources(str(Path(pycaps.__file__).parent / "template" / "preset" / "hype" / "resources"))  # komika.ttf ships with pycaps
    b.with_custom_subtitle_renderer(PictexSubtitleRenderer())
    if len(argv) > 4:  # SETTINGS caption theme: one colour for every active word
        b.add_css_content(f".word-being-narrated, .hype.word-being-narrated {{ color: {argv[4]}; }}")
    b.build().run()


if __name__ == "__main__":
    main(sys.argv[1:])
