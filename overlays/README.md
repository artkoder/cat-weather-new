# Overlay Assets for "Guess the Location"

Place semi-transparent PNG badges in this directory to number quiz photos for the `guess_arch` rubric.

- **File naming:** use `1.png`, `2.png`, … matching the slide order. The bot automatically picks the file that corresponds to the asset's position in the carousel.
- **Transparency:** RGBA/alpha transparency is fully supported. Keep anti-aliased edges and shadows in the PNG—no further work is required.
- **Dimensions:** provide square images at least 512×512 px. The bot rescales overlays to roughly 10–16% of the shorter photo side so they remain legible without hiding too much of the picture.
- **Safe area:** important artwork should stay away from the top-left corner of the source photo. The bot offsets overlays by 24 px (or 12 px on very small photos) to stay clear of Telegram's default crop.

If a numbered PNG is missing, the bot falls back to generating a badge on the fly.
