
import argparse
import struct
import sys
import numpy as np
from scipy import signal
from math import log10
from receiver import butter_bandpass_filter, getDecimate, getNewSigRate, getFrameLength, getFrameRate
from tones import TonesMonitor, TonesRecord,\
    note, printHeader, printFrame, getTimestamp,\
    generateToneTemplate, DebugTonesFormat,\
    TONES

def read_s16le(inp_stream, sig_rate: int):
    
    data = inp_stream.read(sig_rate * 2)
    if not data:
        return []
    
    int_list = []
    idx = 0;
    while idx < len(data):
        
        try:
            two_bytes = data[idx:idx+2]
            value = struct.unpack(f'<h', two_bytes)[0]
            int_list.append(value)
            idx += 2
        except struct.error as e:
            # Handle cases where incomplete data is read at the end
            print(f"Warning: Incomplete 16-bit integer detected at end of input. {e}")
            break
    return int_list

def printTimestamp(format:DebugTonesFormat):
    if (format == DebugTonesFormat.DEBUG_TONES_NONE):
        return
    
    print(f"{getTimestamp()}: ", end='')


def monitor_stream(inp_sig_rate:int, freq_hz: int, 
                   pformat:DebugTonesFormat, 
                   selcal_log_fn: str, 
                   min_group_cnt: int = 3, min_score = 4.5):
    
    decimate = getDecimate(inp_sig_rate)
    sig_rate = getNewSigRate(inp_sig_rate)
    print(f'Selcal detection sensitivity settings -  Min Group Cnt: [{min_group_cnt},  Min Tone Score: [{min_score}].')
    print(f'Input Sample rate {inp_sig_rate}, Decimate: [{decimate}], New SigRate: [{sig_rate}].  Logging events to: [{selcal_log_fn}].')

    # Frames rate/length
    frame_len = getFrameLength(inp_sig_rate)
    frame_rate = getFrameRate(inp_sig_rate)
    template = generateToneTemplate(frame_len, sig_rate)
        
    printTimestamp(pformat)
    printHeader(pformat)
    
    tone_mon = TonesMonitor(freq_hz, selcal_log_fn)

    frameCnt = 0;
    while (True):

        sig_noise = read_s16le(inp_stream=sys.stdin.buffer, sig_rate=inp_sig_rate)
        if (len(sig_noise) == 0):
            break
        
        if decimate > 1:
            sig_noise = signal.decimate(sig_noise, decimate)
            
        sig_noise = butter_bandpass_filter(sig_noise,
                                        270,
                                        1700,
                                        sig_rate,
                                        order=8)

        frames = int(len(sig_noise) / frame_len)
        for frame in range(0, frames):

            beg = frame * frame_len
            end = (frame+1) * frame_len

            corr = np.zeros(len(TONES))

            for tone in range(0, len(TONES)):
                corr[tone] = log10(np.abs(signal.correlate(sig_noise[beg:end],
                                                        template[tone],
                                                        mode='same')).sum())

            tones_rec = TonesRecord(corr)

            printTimestamp(pformat)
            printFrame(frameCnt, tones_rec, pformat)
            frameCnt += 1
            
            # Track Tones:
            #  For TonesByMaxTone min_group_cnt value 3-4 is a good starting level
            #  For TonesByScore min_score 4.5 is a good starting level 
            selcal = tone_mon.trackTones(tones_rec, queue_window_size=frame_rate, min_group_cnt=min_group_cnt, min_score=min_score)

            print(f' - Tone: {selcal["current_tgc"]} - Selcal: [{selcal["selcal"]}] (Act: {selcal["is_active"]}, Q1: {selcal["tg1"]}={selcal["tg1_cnt"]}, Q2: {selcal["tg2"]}={selcal["tg2_cnt"]}) ', flush=True)


def processArgs(parser):

    parser = argparse.ArgumentParser(description="KA9Q-Radio Js8 Decoding Controler.")
    parser.add_argument("-f", "--freq-hz", type=int, help="Frequency (Hz) which feed is streaming from. (For logging purposes)")
    parser.add_argument("-sr", "--sig-rate", type=int, default=11025, choices=[11025, 22050, 44100, 48000], help="Sample rate for raw s16le input.")
    parser.add_argument("-l", "--log", type=str, default="./selcal.log", help="Sample rate for raw s16le input.")
    parser.add_argument("-df", "--debug_fmt", type=str, default="compact", choices=["compact", "max-only", "max+avg"], help="Enable debuging tones and ton levels.")
    parser.add_argument("-mgc", "--min-group-cnt", type=int, default=4, help="Min number of occurences for tone groups to appear in each Q1 & Q2 before Selcal considered active.")
    parser.add_argument("-mts", "--min-tone-score", type=float, default=4.5, help="Min score for all tones before selcal considered active.")
    
    args = parser.parse_args()

    return args        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SELCAL Monitoring & Decoding Utility.")
    args = processArgs(parser)

    pformat = DebugTonesFormat.DEBUG_TONES_NONE
    if (args.debug_fmt == "compact"):
        pformat = DebugTonesFormat.DEBUG_TONES_COMPACT
    elif  (args.debug_fmt == "max-only"):
        pformat = DebugTonesFormat.DEBUG_TONES_MAX_ONLY
    else:
        pformat = DebugTonesFormat.DEBUG_TONES_MAX_AND_ABOVE_AVG

    monitor_stream(args.sig_rate, args.freq_hz, pformat, args.log, min_group_cnt=args.min_group_cnt, min_score=args.min_tone_score)
