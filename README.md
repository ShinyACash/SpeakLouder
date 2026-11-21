# SpeakLouder

sends text / files over the air between two laptops using speakers and mics (ultrasonic 4-fsk around 18-20khz, reed-solomon, crc16, stop-and-wait arq) with a proud range of 4-5m!! wow such usability am I right? I mean if you have bigger speakers and mics you might as well get more range ngl. I mean you can change the frequency range but I'm pretty sure we're already pushing the limits of what laptop hardware can do lol. A fun lil hobby project made out of pure curiosity and the fact that I have way too much time on my hands even though I have Computer Networks End Sem exam showing up (ig this project accounts for me actually studying the syllabus ig.)

## setup

```powershell
pip install -r requirements.txt
```

check your device ids first 'cause windows loves routing audio to your bluetooth headphones instead of the actual speakers:

```powershell
python air.py --devices
```

**important:** turn off "audio enhancements" / noise suppression in windows sound settings for both the mic and speakers, otherwise windows thinks the data is fan noise and deletes it.

## usage

### chat between two pcs

on your friend's pc (listener):
```powershell
python air.py --chat --listen --level 0.6 --band near --verbose
```

on your pc (initiator):
```powershell
python air.py --chat --level 0.6 --band near --verbose
```

turns alternate. type `/quit` to dip.

*(note: default band is `ultrasonic` (18.2-19.4khz). if one of the laptop mics is trash and cuts off above 16khz, `--band near` drops it to 14.8-16.0khz which works way better. there's also `--band audible` if you hate your ears). you may also set --input-device and --output-device to choose which one emits and receives data.* 

### one-way / file transfer

receiver:
```powershell
python air.py --rx --band near --verbose
```

sender:
```powershell
python air.py --tx "bro SPEAK LOUDER" --band near --level 0.6
python air.py --tx-file .\payload.bin --band near --level 0.6  # lowkey doesnt work that well rn
```

if one side doesn't have speakers to send ACKs back, just slap `--no-ack` on both, it'll act as udp and keep spraying packets:
```powershell
python air.py --rx --no-ack --band near
python air.py --tx "hello gng" --no-ack --repeats 2 --band near --level 0.6
```

## testing & range

quick loopback test on one pc:
```powershell
python air.py --hwtest 
```

range test (walks you through distances like 1m, 3m, 5m and dumps snr / fec stats):
```powershell
python air.py --rangetest --level 0.5
```

unit tests / sim:
```powershell
python -m unittest discover -s tests -v
python air.py --sim "test"
```
