def loadspk(spkid):
    import struct

    c = '/home/sohail/data/ns/visual-crcns/crcns_pvc3_cat_recordings/natural_movie/spike_data'
    import os
    os.system('ls '+c+'/*.spk')

    #f1='t00.spk'
    #f1='t00.tem'
    fn='t%02d.spk'%(spkid,)
    ffn=c+'/'+fn


    STEP = 1.0/1000.0/1000.0 # Microseconds
    spkt=[]
    #try:
    print 'Openning '+ffn

    o=file(ffn)
    while True:
        #for j in range(0,5):
        b8=o.read(8)
        if len(b8)==0:
            #end of file
            break;
        d=struct.unpack('Q',b8)
        #%print type(d)
        #%print len(d)
        #%print type(d[0])
        st = float(d[0]+0.01) * STEP * 1000.0  /1000 #msec

        million=1000000.0
        st = round(st *million)/million

        #    print st
        #if d.
        spkt.append(st)
        #print float(len(spkt)) / st #spikes per secnd
    #1pm--4am
    #-1hr
    #14hrs?   14*15 = 7*2*5*3=210 GBP
    #6-7 spikes per sec?
    if len(spkt)>0:
        return spkt

    #except Exception as e:  #end of file
    #    print e
    #    if len(spkt)>0:
    #        return spkt
    raise Exception("SHould never reach here")


if __name__ == "__main__":
    spktrain=loadspk(00)
    #print spktrain
    CHA=[0,2,4,8,10,18,23,25,26,27]
    yi=0
    for spkid in CHA: #range(0,15):
        yi+=1
        from  spkloader import loadspk
        a=loadspk(spkid)
        import matplotlib.pylab as pl
        import numpy
        pl.plot(a,numpy.random.rand(len(a))*0.7  + (yi-1)*1.0,'.')
    pl.ylim([-1,len(CHA)+1])
    pl.show()



"""
history:
from  spkloader import loadspk
a=loadspk(0)
a
import matplotlib as p
import matplotlib.pylab as pl
pl.xkcd
help(pl.xkcd)
pl.plot(a)
pl.plot(a)
pl.show()
pl.show()
pl.plot(a,)
import numpy
pl.plot(a,numpy.random.rand((len(a),)) )
z=numpy.random.rand((len(a),))
z=numpy.random.rand((len(a),))
z=numpy.random.rand(len(a))
z=numpy.random.rand(len(a))
len(z)
help(numpy.random.rand(len(a)))
help(numpy.random.rand)
help(numpy.random.rand(len(a)))
pl.plot(a,numpy.random.rand(len(a)) )
pl.show()
pl.cla()
pl.show()
pl.plot(a,numpy.random.rand(len(a)) )
pl.show()
pl.cla()
pl.plot(a,numpy.random.rand(len(a)) ,'.')
pl.show()
pl.ylim([-3,10])
pl.show()
pl.plot(a,numpy.random.rand(len(a)) ,'.')
pl.show()
pl.show()
pl.show()
pl.plot(a,numpy.random.rand(len(a)) ,'.')
pl.ylim([-3,10])
pl.show()
history

"""
