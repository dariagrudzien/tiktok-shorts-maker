# tiktok-shorts-maker
Documentation TBA

How to run:

1. Set your [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) to avoid rate-limiting:

```
export HF_TOKEN=<token>
```

2. Create the short:

```
python3 create_shorts.py \
  --input "input-video.mp4" --outdir "./shorts_out" \
  --num-shorts 25 --min-sec 2 --max-sec 35 --min-gap 0.4 \
  --caption-preset tiktok \
  --fontsize 108 --margin-h 220 --center-offset 0 \
  --chunk-gap 0.12 --theme white-blue \
  --box-opacity 0 \
  --prefer faster --model medium --device auto
```

3. Add subtitles to the whole video instead of picking shorts (with a review/edit
   step in between):

```
# Step 1: transcribe and write outdir/subtitles.srt for you to review/edit
python3 create_shorts.py --input "input-video.mp4" --outdir "./subs_out" --mode subtitles

# Edit ./subs_out/subtitles.srt as needed, then render the final video:
python3 create_shorts.py --input "input-video.mp4" --outdir "./subs_out" \
  --mode subtitles --srt "./subs_out/subtitles.srt" \
  --caption-preset tiktok --theme white-blue
```

How to compile video with subs:
```
ffmpeg -hide_banner -loglevel error -y \
  -ss 142.38 -i "input-video.mp4" -t 33.30 \
  -vf "crop=608:1080:607:0,scale=1080:1920,ass='./shorts_out/input.ass'" \
  -r 30 -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p \
  -c:a aac -b:a 128k "./shorts_out/input_fixed.mp4"
```
