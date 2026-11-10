# Codec/processing ladder: encode each POC cut's audio, decode back to 48 kHz mono wav, run the blind attack.
set -u
REPO=${REPO:-$HOME/src/flyover-scrub}
cd "${WORK:-.}"  # run from a work dir holding aud/<clip>.wav + .meta and the flight-log dirs
mkdir -p ladder
declare -a V=(
 "ref|-c:a pcm_s16le|wav"
 "aac_apple_128k|-c:a aac_at -b:a 128k|m4a"
 "aac_ffmpeg_128k|-c:a aac -b:a 128k|m4a"
 "aac_ffmpeg_64k|-c:a aac -b:a 64k|m4a"
 "aac_ffmpeg_32k|-c:a aac -b:a 32k|m4a"
 "he_aac_apple_48k|-c:a aac_at -profile:a 4 -b:a 48k|m4a"
 "he_aac_ffmpeg_64k_instagramlike|-c:a aac -profile:a aac_he -b:a 64k|m4a"
 "opus_128k|-c:a libopus -b:a 128k|ogg"
 "opus_32k|-c:a libopus -b:a 32k|ogg"
 "opus_16k_voip|-c:a libopus -b:a 16k -application voip|ogg"
 "opus_8k_voip|-c:a libopus -b:a 8k -application voip|ogg"
 "mp3_64k|-c:a libmp3lame -b:a 64k|mp3"
 "phone_amrnb_12k|-ar 8000 -c:a libopencore_amrnb -b:a 12.2k|amr"
 "wideband_g722|-ar 16000 -c:a g722|wav"
 "hp100_aac128|-af highpass=f=100:poles=2 -c:a aac -b:a 128k|m4a"
 "hp200_aac128|-af highpass=f=200:poles=2 -c:a aac -b:a 128k|m4a"
 "loudnorm14_aac128|-af loudnorm=I=-14:TP=-1 -c:a aac -b:a 128k|m4a"
 "agc_dynaudnorm_aac128|-af dynaudnorm=f=150:g=15 -c:a aac -b:a 128k|m4a"
)
for clip in IMG_2173_cut IMG_4367_cut; do
  for v in "${V[@]}"; do
    IFS='|' read name opts ext <<< "$v"
    out=${clip}__$name
    [ -f ladder/$out.done ] && continue
    ffmpeg -nostdin -loglevel error -y -i aud/$clip.wav -ac 1 $opts ladder/tmp.$ext && \
    ffmpeg -nostdin -loglevel error -y -i ladder/tmp.$ext -ac 1 -ar 48000 aud/$out.wav && cp aud/$clip.meta aud/$out.meta || { echo "$out ENCODE-FAIL"; continue; }
    if [ $clip = IMG_2173_cut ]; then L=$(ls full2173/*.jsonl | tr '\n' ',' | sed 's/,$//'); args="10 75 65 95 5000"; F=""; else L=$(ls full4367/*.jsonl | tr '\n' ',' | sed 's/,$//'); args="33 68 55 84 5000"; F=1; fi
    r=$(FOLLOW=$F NOSCRUB=1 uv run -q --project "$REPO" --with matplotlib python trace_attack_research.py $out $L $args 2>&1 | grep -E "best flight|^original" | tr '\n' ' ')
    echo "$out :: $r" | tee -a ladder/results.txt; touch ladder/$out.done; rm -f aud/$out.wav
  done
done
echo LADDER-FINISHED
