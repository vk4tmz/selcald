#!/usr/bin/bash

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

sox $1 -t raw -r $2 -b 16 -c 1 - |  python ./selcal_monitor.py -sr ${2-11025} -f 17904000 -df ${3-compact}

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ""

python ./receiver.py $1

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
