from flask import Flask
import sysv_ipc 
import threading
import psutil
import time

bws = []
NETWORK_INTERFACE = 'eno1'
bw = 0
memory = sysv_ipc.SharedMemory(6652, flags=sysv_ipc.IPC_CREAT, mode=0o666, size=10)
class Worker(threading.Thread):
    def __init__(self, name):
        super().__init__()
        self.name = name            # thread 이름 지정

    def run(self):
        global bws
        global bw
        previous = 0
        while(1):
            time.sleep(1)
            memory.attach()
            netio = psutil.net_io_counters(pernic=True)
            net_usage = netio[NETWORK_INTERFACE].bytes_sent
            net_usage = netio[NETWORK_INTERFACE].bytes_recv
            bw = 1000 - ((net_usage-previous) * 8 / 1024 / 1024)
            if (bw <0):
                print("???", bw)
                previous = net_usage 
                continue
            bw = bw;
            if len(bws) == 10:
                del bws[0]
                bws.append(bw)
            else:
                bws.append(bw)
            bw = sum(bws)/len(bws)
            previous = net_usage
            memory.write("0000000000")
            zero = "0" * (10 - len(str(int(bw))))
            memory.write(zero + str(int(bw)))
            memory.detach()
            #memory.remove()

app = Flask(__name__)

class CShmReader : 
    def __init__( self, key ) : 
        self.memory = sysv_ipc.SharedMemory( key ) 
        pass 
    def doReadShm( self ) : 
        self.memory.attach()
        memory_value = self.memory.read()
        self.memory.detach()
        return  memory_value
    def doDetach(self):
        self.memory.detach()
s = CShmReader(1991)

@app.route('/status')
def get_status():
    start = time.time()*1000000
    current_time = str(int(time.time()*1000000))
    ret = s.doReadShm( )
    ret = ret.decode('ascii').split("\n")[0]
    ret = ret.replace("*","")
    kkk = ret.split(",")
    if(len(kkk) == 7):
        for i in range(9):
            ret+="0,"
        ret+="0"
    print(ret)
    #s.doDetach()
    end = time.time()*1000000
    return current_time + "," + ret + "," +str(int(bw))+","+str(int(end-start))
    
import requests
if __name__== '__main__':

    th1 = Worker("network")
    th1.daemon = True
    th1.start()
    app.run(debug=True, host='0.0.0.0',port=8004, threaded=True)
