"""Captions + sound cues on a rendered 9:16 short with pycaps (github.com/francozanardi/pycaps, MIT):

  python editor/captions.py <in.mp4> <out.mp4> <captions.json> <y 0..1> <pop|story> [#active-word colour]

captions.json = whisper_json {"segments": [{"start", "end", "words": [{"word", "start", "end"}]}]} in OUTPUT seconds
(renderer.caption_doc: Whisper on the cut audio, so no drift). y = where the caption block's centre sits (fraction of
the height: renderer.CAPTION_Y per layout). Styles: pop = editor/pycaps_style (bold uppercase Komika Axis, active
word zoom-in + yellow, hype words green / red, a soft pop cue on hype words); story = editor/pycaps_style_story (the
emotional_story profile: smaller sentence-case Archivo Black in white / soft grey, soft shadow, a gentle fade, no
tags, colours or cues). Both fonts ship inside pycaps (no system fonts needed on the Pod / mini PC). Renders without a browser
(PictexSubtitleRenderer) and never calls an AI service (no "ai" tagger rules, no emoji effects).
renderer.burn_captions runs this as its own process: pycaps / movielite print progress bars and may spawn workers.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STYLES = {"pop": (HERE / "pycaps_style" / "pycaps.template.json", "hype"),  # (template, pycaps preset with its font)
          "story": (HERE / "pycaps_style_story" / "pycaps.template.json", "default")}


def main(argv):
    import pycaps
    from pycaps import JsonConfigLoader, PictexSubtitleRenderer
    from pycaps.layout import SubtitleLayoutOptions

    src, out, doc, y = argv[0], argv[1], json.loads(Path(argv[2]).read_text(encoding="utf-8")), float(argv[3])
    style, preset = STYLES[argv[4] if len(argv) > 4 else "pop"]
    b = JsonConfigLoader(str(style)).load(False)
    b.with_input_video(src).with_output_video(out).with_transcription(doc).should_save_subtitle_data(False)
    # pycaps' "center" puts the block's top at (H - h) * (offset + 0.5): offset y - 0.5 centres it at ~y
    layout = json.loads(style.read_text(encoding="utf-8"))["layout"]
    b.with_layout_options(SubtitleLayoutOptions(**dict(layout, vertical_align={"align": "center", "offset": y - 0.5})))
    b.with_resources(str(Path(pycaps.__file__).parent / "template" / "preset" / preset / "resources"))  # the font ships with pycaps
    b.with_custom_subtitle_renderer(PictexSubtitleRenderer())
    if len(argv) > 5 and style == STYLES["story"][0]:  # the emphasis line's accent colour + its glow ring
        b.add_css_content(accent_css(argv[5]))
    elif len(argv) > 5:  # SETTINGS caption theme: one colour for every active word (pop only)
        b.add_css_content(f".word-being-narrated, .hype.word-being-narrated {{ color: {argv[5]}; }}")
    b.build().run()


def accent_css(color):
    """#RRGGBB -> the story style's .last-line-in-segment colour + a glow ring of hard shadows in that colour."""
    r, g, b = (int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    ring = ", ".join(f"{x}px {y}px 0 rgba({r},{g},{b},0.38)" for x, y in ((-2.2, 0), (2.2, 0), (0, -2.2), (0, 2.2)))
    return (f".last-line-in-segment {{ color: {color}; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, "
            f"1px 1px 0 #000, {ring}, 1.5px 2px 0 rgba(0,0,0,0.6); }}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
