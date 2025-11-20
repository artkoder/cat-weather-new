-- Scale postcard_score to the new 10-point range.
UPDATE assets
SET postcard_score = postcard_score * 2
WHERE postcard_score IS NOT NULL;
