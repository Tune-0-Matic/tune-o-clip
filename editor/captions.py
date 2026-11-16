"""Captions + sound cues on a rendered 9:16 short with pycaps (github.com/francozanardi/pycaps, MIT):

  python editor/captions.py <in.mp4> <out.mp4> <captions.json> [#active-word colour]

captions.json = whisper_json {"segments": [{"start", "end", "words": [{"word", "start", "end"}]}]} in OUTPUT seconds
(renderer.caption_doc: Whisper on the cut audio, so no drift). Style: editor/pycaps_style (bold uppercase Komika Axis,
active word pop + yellow, hype words green / red, a soft pop cue on hype words). Renders without a browser
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

    src, out, doc = argv[0], argv[1], json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    b = JsonConfigLoader(str(STYLE)).load(False)
    b.with_input_video(src).with_output_video(out).with_transcription(doc).should_save_subtitle_data(False)
    b.with_resources(str(Path(pycaps.__file__).parent / "template" / "preset" / "hype" / "resources"))  # komika.ttf ships with pycaps
    b.with_custom_subtitle_renderer(PictexSubtitleRenderer())
    if len(argv) > 3:  # SETTINGS caption theme: one colour for every active word
        b.add_css_content(f".word-being-narrated, .hype.word-being-narrated {{ color: {argv[3]}; }}")
    b.build().run()


if __name__ == "__main__":
    main(sys.argv[1:])
