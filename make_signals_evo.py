import numpy
import numpy.random

def make_lor1(ntr,nlen):
    a = numpy.random.randn(ntr,nlen)
    b= numpy.cumsum(a, axis=1 )

    #import plotdata
    #import matplotlib.pyplot as pyplot
    #import plotdata
    import matplotlib.pyplot
    #matplotlib.pyplot.plot(a[0,:])
    #z=matplotlib.pyplot.plot(a[0,:])
    z=matplotlib.pyplot.plot(a[0,0:500])
    z2=matplotlib.pyplot.plot(b[0,0:500])
    gfn='graph3.png'
    matplotlib.pyplot.savefig(gfn)
    import os
    os.system("dolphin %s" % gfn)


#main
ntr=100
nlen=100000
make_lor1(ntr,nlen)
