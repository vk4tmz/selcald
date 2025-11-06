
import numpy as np

from collections import deque
from datetime import datetime, timezone
from enum import Enum


TONES = [312.6,
         346.7,
         384.6,
         426.6,
         473.2,
         524.8,
         582.1,
         645.7,
         716.1,
         794.3,
         881.0,
         977.2,
         1083.9,
         1202.3,
         1333.5,
         1479.1]

ALPHABET = ['Alpha',
            'Bravo',
            'Charlie',
            'Delta',
            'Echo',
            'Foxtrot',
            'Golf',
            'Hotel',
            'Juliette',
            'Kilo',
            'Lima',
            'Mike',
            'Papa',
            'Quebec',
            'Romeo',
            'Sierra']


#####################################################################################
# Class TonesMonitor - 
#   - Manages 2 queues that holds the set of Tone Group codes decoded per frame.
#     Q1 - Will contain tone group code(s) assessed for potential first part of Selcal
#     Q2 - Will contain tone group code(s) assessed for potential second part of Selcal
#
#  - Each queue holds all the tone group codes from each frame for 1 second each (ie 2 seconds all up)
#  - As a new frame is decoded, its pushed onto the bottom of Q2, once Q2 if full it pops the oldest
#    tone group code from the top of Q@ and pushes onto bottom of Q1, Once Q1 is full the oldest entry is
#    popped from its top of the queue
#  - We then determine for each queue the tone group with the highest count.  Tone Group code highest count 
#    in Q2 is excluded from Q1's set to ensure 2 different TGC
#  - Once both queues have a tone group count >= than specified threshold, then selcal is deemed active.
#
#####################################################################################

class TonesRecord:

    score_bin_size:int=5
    scores:list
    corr:list
    max1idx:int
    max2idx:int
    max:float
    min:float
    avg:float

    gtc:str=None

    def __init__(self, corr:list):
        self.corr = corr
        self.computeStats()

    # Computes:
    #  - Min,Max and Average for the Corr set
    #  - Top 2 tones
    def computeStats(self):
        max1 = 0.0
        idx1 = -1
        max2 = 0.0
        idx2 = -1
        tot = 0.0
        for tone in range(0, len(TONES)):
            tot += self.corr[tone]
            if (self.corr[tone] > max1):
                max1 = self.corr[tone]
                idx1 = tone       

        for tone in range(0, len(TONES)):
            if ((tone != idx1) and (self.corr[tone] > max2)):
                # Only change max2 if delta its greater than that of 1/4th delta between itself and max1
                # (ie needs to be a considerable change is max), if not stick with the first
                if ((self.corr[tone] - max2) > ((max1 - max2) / 4)):
                    max2 = self.corr[tone]
                    idx2 = tone  

        if (idx1>idx2):
            t = idx1
            idx1 = idx2
            idx2 = t

        self.max1idx = idx1
        self.max2idx = idx2
        self.max = max1
        self.avg = tot / len(TONES)
        
        # Tone as single uppercase alphabet character
        t1_c = ALPHABET[idx1][:1]
        t2_c = ALPHABET[idx2][:1]

        # Build the current group tone code digraph
        self.gtc = t1_c + t2_c

        self.computeScores()


    def computeScores(self):
        bin_score = 1 / self.score_bin_size
        bin_corr = (self.max - self.avg) / self.score_bin_size

        self.scores = [0] * len(TONES)
        for tone in range(0, len(TONES)):
            cv = self.corr[tone]

            # if cv is one of the MAX tones weight=1
            if ((tone == self.max1idx) or (tone == self.max2idx)):
                self.scores[tone] = 1.0

            # Only score those above average, other remaining get 0
            elif (cv > self.avg):
                #dv = self.max - cv
                dv = cv - self.avg
                binIdx = int(dv / bin_corr)
                self.scores[tone] = round(binIdx * bin_score, 1)
        
        #print(f" Scores: [{self.scores}] ", end='')


class TonesMonitor:
    # Track
    tonesQ1 = deque()
    tonesQ2 = deque()
    tonesCnt1 = {}
    tonesCnt2 = {}

    tonesQ1Score = [0] * len(TONES)
    tonesQ2Score = [0] * len(TONES)
    tonesQ1MaxCnt = [0] * len(TONES)
    tonesQ2MaxCnt = [0] * len(TONES)

    lastSelcall = []
    lastSelcall_BS = []

    selcall_log_fn: str
    freq_hz : int

    def __init__(self, freq_hz:int, selcall_log_fn: str="./selcal.log"):
        self.freq_hz = freq_hz
        self.selcall_log_fn = selcall_log_fn

    def incCounter(self, tonesQ:dict, gtc:str):
        if gtc not in tonesQ:
            tonesQ[gtc] = 1
        else:
            tonesQ[gtc] += 1

    def decCounter(self, tonesQ:dict, gtc:str):
        if gtc in tonesQ:
            tonesQ[gtc] -= 1

    def resetToneScores(self):
        self.tonesQ1Score = [0] * len(TONES)
        self.tonesQ2Score = [0] * len(TONES)
        self.tonesQ1MaxCnt = [0] * len(TONES)
        self.tonesQ1MaxCnt = [0] * len(TONES)

    def incScores(self, toneQScores:list, tonesQMaxCnt:list, trec:TonesRecord):
        for tone in range(0, len(TONES)):
            toneQScores[tone] += trec.scores[tone]

        tonesQMaxCnt[trec.max1idx] += 1
        tonesQMaxCnt[trec.max2idx] += 1

    def decScores(self, toneQScores:list,  tonesQMaxCnt:list, trec:TonesRecord):
        for tone in range(0, len(TONES)):
            if (toneQScores[tone] > 0):
                toneQScores[tone] -= trec.scores[tone]
            
        tonesQMaxCnt[trec.max1idx] += 1
        tonesQMaxCnt[trec.max2idx] += 1                

    def trackByMaxTones(self, trec:TonesRecord, queue_window_size:int, min_group_cnt:int, res:dict):
        q1_max_tgc = None
        q1_max_tgc_cnt = 0
        q2_max_tgc = None
        q2_max_tgc_cnt = 0

        #self.tonesQ2.append(trec.gtc)
        self.tonesQ2.append(trec)
        self.incCounter(self.tonesCnt2, trec.gtc)
        self.incScores(self.tonesQ2Score, self.tonesQ2MaxCnt, trec)
        
        # Q1 is latest (will be the 2nd part), Q2 oldest (will be the first part)
        if (len(self.tonesQ2) > queue_window_size):         

            # pop of last Tone Record in queue and reduce its count
            last_trec = self.tonesQ2.popleft()
            last_gtcQ2 = last_trec.gtc
            self.decCounter(self.tonesCnt2, last_gtcQ2)
            self.decScores(self.tonesQ2Score,self.tonesQ2MaxCnt, last_trec)
                    
            # Append and increase cnt
            self.tonesQ1.append(last_trec)
            self.incCounter(self.tonesCnt1, last_trec.gtc)
            self.incScores(self.tonesQ1Score, self.tonesQ1MaxCnt, last_trec)

            if (len(self.tonesQ1) > queue_window_size):
                last_trecQ1 = self.tonesQ1.popleft()
                last_gtcQ1 = last_trecQ1.gtc
                self.decCounter(self.tonesCnt1, last_gtcQ1)
                self.decScores(self.tonesQ1Score, self.tonesQ1MaxCnt, last_trecQ1)

        # Lets scan the sliding windows and determine the TGC with highest cnt in each half
        for q2_tgc in self.tonesCnt2.keys():
            cnt = self.tonesCnt2[q2_tgc]
            if (cnt > q2_max_tgc_cnt):
                q2_max_tgc = q2_tgc
                q2_max_tgc_cnt = cnt

        for q1_tgc in self.tonesCnt1.keys():
            cnt = self.tonesCnt1[q1_tgc]
            # Only cnt in max logic if not the same code from Q2
            if (q1_tgc != q2_max_tgc) and (cnt > q1_max_tgc_cnt):
                q1_max_tgc = q1_tgc
                q1_max_tgc_cnt = cnt
                
        # Have we encounter active selcall ? 
        if ((q1_max_tgc_cnt >= min_group_cnt) and
            (q2_max_tgc_cnt >= min_group_cnt) and
            (q1_max_tgc != q2_max_tgc)): 
            res['is_active'] = True
            res['selcal'] = f"{q1_max_tgc}-{q2_max_tgc}"

            # Confirmed selcall active
            if (len(self.lastSelcall) == 0):
                self.lastSelcall = [q1_max_tgc, q2_max_tgc]

                # log selcal event
                selcall_event = f"{getTimestamp()} {self.freq_hz/1000:.01f} kHz {res['selcal']} ~ SELCAL_BYMAXTONE\n"
                writeStringToFile(self.selcall_log_fn, selcall_event)

        else:
            res['is_active'] = False
            res['selcal'] = None

            # Looks like end of prev selcall, Clear both all counters
            if (len(self.lastSelcall) > 0):                
                self.tonesCnt1 = {}
                self.tonesCnt2 = {}

                self.lastSelcall = []
        
        # Update Q1 & Q2 stats
        res['tg1'] = q1_max_tgc
        res['tg1_cnt'] = q1_max_tgc_cnt
        res['tg2'] = q2_max_tgc
        res['tg2_cnt'] = q2_max_tgc_cnt

    # min_score = 4.5 - Good start minimal false decodes
    def trackByScore(self, trec:TonesRecord, queue_window_size:int, min_score:float, res:dict):
        top2Q1 = top2(self.tonesQ1Score)
        top2Q2 = top2(self.tonesQ2Score, top2Q1["idx"])

        q1Max1idx = top2Q1["idx"][0]
        q1Max2idx = top2Q1["idx"][1]
        q1Max1Cnt = self.tonesQ1MaxCnt[q1Max1idx]
        q1Max2Cnt = self.tonesQ1MaxCnt[q1Max2idx]
        q1Max1Val = top2Q1["val"][0]
        q1Max2Val = top2Q1["val"][1]

        q2Max1idx = top2Q2["idx"][0]
        q2Max2idx = top2Q2["idx"][1]
        q2Max1Cnt = self.tonesQ2MaxCnt[q2Max1idx]
        q2Max2Cnt = self.tonesQ2MaxCnt[q2Max2idx]
        q2Max1Val = top2Q2["val"][0]
        q2Max2Val = top2Q2["val"][1]

        # DEBUG
        #print(f" [ Score: (Q1: {q1Max1idx}:{q1Max1Cnt}, {q1Max2idx}:{q1Max2Cnt})  Q2: ({q2Max1idx}:{q2Max1Cnt}, {q2Max2idx}:{q2Max2Cnt}) ] ", end='')
        #print(f" [ Score: (Q1: {q1Max1idx}:{q1Max1Val:.01f}, {q1Max2idx}:{q1Max2Val:.01f})  Q2: ({q2Max1idx}:{q2Max1Val:.01f}, {q2Max2idx}:{q2Max2Val:.01f}) ] ", end='')
        print(f" [ Score: (Q1: {q1Max1idx}:{q1Max1Val:.01f}, {q1Max2idx}:{q1Max2Val:.01f})  Q2: ({q2Max1idx}:{q2Max1Val:.01f}, {q2Max2idx}:{q2Max2Val:.01f}) ] ", end='')
        
        if ((q1Max1Val >= min_score) and
            (q1Max2Val >= min_score) and
            (q2Max1Val >= min_score) and
            (q2Max2Val >= min_score)):        

            # Confirm Q1 and Q2 tgc are not overlapping
            if ((q1Max1idx != q2Max1idx) and 
                (q1Max1idx != q2Max2idx) and
                (q1Max2idx != q2Max1idx) and
                (q1Max2idx != q2Max2idx)):

                q1_score_tgc = ALPHABET[top2Q1["idx"][0]][:1] + ALPHABET[top2Q1["idx"][1]][:1]
                q2_score_tgc = ALPHABET[top2Q2["idx"][0]][:1] + ALPHABET[top2Q2["idx"][1]][:1]

                res['is_active_BS'] = True
                res['selcal_BS'] = f"{q1_score_tgc}-{q2_score_tgc}"

                # Confirmed selcall active
                if (len(self.lastSelcall_BS) == 0):
                    self.lastSelcall_BS = [q1_score_tgc, q2_score_tgc]
                
                    selcal_byscore = f"{q1_score_tgc}-{q2_score_tgc}"

                    # log selcal event
                    stats = f" [ Score: (Q1: {q1Max1idx}:{q1Max1Val:.01f}, {q1Max2idx}:{q1Max2Val:.01f})  Q2: ({q2Max1idx}:{q2Max1Val:.01f}, {q2Max2idx}:{q2Max2Val:.01f}) ] "
                    selcall_event = f"{getTimestamp()} {self.freq_hz/1000:.01f} kHz {selcal_byscore} ~ SELCAL_BYSCORE - STATS: [{stats}]\n"
                    writeStringToFile(self.selcall_log_fn, selcall_event)                   

        else:
            
            res['is_active_BS'] = False
            res['selcal_BS'] = None

            # Looks like end of prev selcall, Clear both all counters
            if (len(self.lastSelcall_BS) > 0):
                self.resetToneScores()
                self.lastSelcall_BS = []

        print(f" SC: [{res['selcal_BS']}] ", end='')

    # 
    # We track 2 sliding windows tonesQ1 (first half / encountered) and tonesQ2 (second flat / encounter)
    #
    def trackTones(self, trec:TonesRecord, queue_window_size: int, min_group_cnt: int = 3, min_score = 4.5):
        
        res = {"current_tgc": trec.gtc,
                "is_active": False,
                "selcal": None, 
                "selcal_BS": None, 
                "tg1": None, "tg1_cnt": 0, 
                "tg2": None, "tg2_cnt": 0, 
        }

        self.trackByMaxTones(trec, queue_window_size, min_group_cnt, res)

        self.trackByScore(trec, queue_window_size, min_score, res)        

        return res
    
#####################################################################################
# Debug and Helper functions for TONES
#####################################################################################

def getTimestamp(): 
    ts = datetime.now(timezone.utc)
    return ts.strftime("%Y/%m/%d-%H:%M:%S")
    

def writeStringToFile(out_fn: str, item: str, append: bool=True):
    wmode = "w"
    if append:
        wmode ="a"

    with open(out_fn, wmode) as file:                    
        file.write(item)

    return 0

class DebugTonesFormat(Enum):
    DEBUG_TONES_NONE = 0
    DEBUG_TONES_COMPACT = 1
    DEBUG_TONES_MAX_ONLY = 2
    DEBUG_TONES_MAX_AND_ABOVE_AVG = 3

# tone synthesis
def note(freq, cycles, amp=32767.0, rate=44100):
    print(f"DEBUG: note() - freq: [{freq}], cycles: [{cycles}], amp: [{amp}], rate: [{rate}]")
    len = cycles * (1.0/rate)
    t = np.linspace(0, len, int(len * rate))
    if freq == 0.0:
        data = np.zeros(int(len * rate))
    else:
        data = np.sin(2 * np.pi * freq * t) * amp
    return data.astype(int)

def generateToneTemplate(frame_len: int, sig_rate: int):
    template = []
    for tone in range(0, len(TONES)):
        template.append(note(TONES[tone], frame_len, rate=sig_rate))

    return template

def printHeader(format:DebugTonesFormat):
    if (format == DebugTonesFormat.DEBUG_TONES_NONE):
        return
    
    if (format == DebugTonesFormat.DEBUG_TONES_COMPACT):
        print(' Index   A  B  C  D  E  F  G', end='')
        print('  H  J  K  L  M  P  Q  R', end='')
        print('  S   Avg')
    else:
        print(' Index     A      B      C      D      E      F      G', end='')
        print('      H      J      K      L      M      P      Q      R', end='')
        print('      S     Avg')


def printSymbol(sym, format:DebugTonesFormat):
    pad = ""
    if (format in [DebugTonesFormat.DEBUG_TONES_MAX_AND_ABOVE_AVG, DebugTonesFormat.DEBUG_TONES_MAX_ONLY]):
        pad = "  "

    print(f" {pad}{sym}{pad} ", end='')


def printValue(val_type, val, format:DebugTonesFormat):
    if (val_type == "MAX"):
        if ((format in [DebugTonesFormat.DEBUG_TONES_MAX_AND_ABOVE_AVG, DebugTonesFormat.DEBUG_TONES_MAX_ONLY])):
            print(f"[{val:5.02f}]", end='')
        else:
            printSymbol("|", format)

    elif (val_type == "AVG") :
        if ((format in [DebugTonesFormat.DEBUG_TONES_MAX_AND_ABOVE_AVG])):
            print(f" {val:5.02f} ", end='')
        else:
            printSymbol("+", format)
    
    else:
        printSymbol(".", format)


def printFrame(frame:int, trec:TonesRecord, format:DebugTonesFormat):
    if (format == DebugTonesFormat.DEBUG_TONES_NONE):
        return
    
    print(f'{frame:06d}: ', end='')
    for tone in range(0, len(TONES)):
        if tone == trec.max1idx or tone == trec.max2idx:                        
            printValue("MAX", trec.corr[tone], format)
        else:
            if trec.corr[tone] > trec.avg:
                printValue("AVG", trec.corr[tone], format)
            else:
                printValue("<=AVG", trec.corr[tone], format)
    
    print(f' {trec.avg:5.02f}', end='')


def top2(vals:list, excluded_idx:list=[]):
    idx1 = -1
    max1 = -1
    idx2 = -1
    max2 = -1
    for tone in range(0, len(vals)):
        if (tone in excluded_idx):
            continue

        if vals[tone] > max1:
            if (idx1 >= 0):
                max2=max1
                idx2=idx1

            max1 = vals[tone]
            idx1 = tone           

        elif vals[tone] > max2:
            max2 = vals[tone]
            idx2 = tone      

    # Ensure order of indexes from smalled to large (NOT value, just index)
    if (idx1>idx2):
        t = idx1
        idx1 = idx2
        idx2 = t

    return {"idx": [idx1, idx2], "val": [vals[idx1], vals[idx2]]}