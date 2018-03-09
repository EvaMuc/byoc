import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm,Normalize
import pickle as pkl
from obspy import UTCDateTime
import os,sys
from pyproj import Proj
import subprocess
from scipy import linalg
from scipy.optimize import lsq_linear


def lonlat2xy(lon,lat,proj=Proj('+proj=utm +zone=37N +datum=WGS84')):
    lat0=8.32
    lon0=38.9
    x0,y0=proj(lon0,lat0)
    x0/=1000
    y0/=1000
    x,y=proj(lon,lat)
    return (x/1000)-x0,y0-(y/1000)
def xy2lonlat(x,y,proj=Proj('+proj=utm +zone=37N +datum=WGS84')):
        lat0=8.32
        lon0=38.9
        x0,y0=proj(lon0,lat0)
        x0/=1000
        y0/=1000
        x=x+x0
        y=y0-y
        x*=1000
        y*=1000
        lon,lat=proj(x,y,inverse=True)
        return lon,lat

plt.close('all')

def get_model(A, y, lamb=0,regularization='ridge',bounds=False,verbose=0):
    n_col = A.shape[1]
    
    if isinstance(regularization,str):
        if regularization == 'ridge':
            # 
            L=np.identity(n_col)
        elif regularization == 'smooth':
            # fill with a finite difference matrix
            L=np.identity(n_col)*2+np.diag(-np.ones(n_col-1),k=1)+np.diag(-np.ones(n_col-1),k=-1)
    else:
        L=regularization
        
    if not bounds:
        return linalg.solve(A.T.dot(A) + lamb * L, A.T.dot(y))
    else:
        lower_bnd=np.ones(n_unknowns)
        upper_bnd=np.ones_like(lower_bnd)
        lower_bnd[:nev+n_stations]*=-np.inf
        lower_bnd[nev+n_stations:]=0
        upper_bnd*=np.inf

        return lsq_linear(A.T.dot(A) + lamb * L, A.T.dot(y),(lower_bnd,upper_bnd),verbose=verbose)

def calculate_smoothing(fun,x):
    s=0
    i=1
    while i < len(fun)-1:
        h=np.asarray([x[i]-x[i-1],x[i+1]-x[i]]).mean()
#        print(fun[i-1]-2*fun[i]+fun[i+1])
        s+=(fun[i-1]-2*fun[i]+fun[i+1])/h**2
        i+=1
    return s

## calculate the FI for each observation getting information about epicentral distance etc
def calculate_FI(fname,edic,comps=['N','E'],reftime=UTCDateTime(2016,1,1),good_stations=[],f_lo=(.6,1.2),f_hi=(6,12)):
    """
    calculate_FI: <pickle file output from ./calc_FI.py> <dictionary of event data>
    """
    
    with open(fname,'rb') as fid:
        FI=pkl.load(fid)
    
    dist=[]
    depth=[]
    fi=[]
    station=[]
    event=[]
    lat=[]
    lon=[]
    time=[]
    for uid in FI.keys():
        for s in FI[uid]['distance'].keys():
    
            if s == 'distance':
                continue
            if s == 'LF':
                continue
            for comp in comps:
                edat=edic[uid]
                stat=s[0:4]
                if stat in good_stations:
                    pass
                else:
                    continue
        
                spectra=FI[uid][s+'-'+comp+'_spectra'][2]
                freq=spectra[1]
                amp=spectra[0]
        
                freq_ind=-np.log10(amp[(freq>=f_lo[0]) & (freq<=f_lo[1])].mean()/amp[(freq>=f_hi[0]) & (freq<=f_hi[1])].mean())
        
                fi.append(freq_ind)
                dist.append(FI[uid]['distance'][s])
                hypdist=np.sqrt(FI[uid]['distance'][s]**2+edat[3]**2)
        
                depth.append(edat[3])
                lat.append(edat[1])
                lon.append(edat[2])
                time.append((edat[0]-reftime)/86400)
                station.append(stat+comp)
                event.append(uid)

    depth=np.asarray(depth)
    dist=np.asarray(dist)
    fi=np.asarray(fi)
    hypdist=np.sqrt(np.power(depth+1.6,2)+np.power(dist,2))
    station=np.asarray(station)
    lat=np.asarray(lat)
    lon=np.asarray(lon)
    time=np.asarray(time)
    event = np.asarray(event)
    return fi,(dist,hypdist),(station,event),(time,lat,lon,depth)

def read_events(fnames):
    dic={}
    for fname in fnames:
        
        # read events
        with open(fname,'r') as fid:
            lines=fid.readlines()

        for line in lines:
            line=line.split()
            uid=UTCDateTime(line[1]).strftime('%Y%j%H%M%S')
            # uid = otime,lat,lon,dep,mag
            dic[uid]=[UTCDateTime(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5])]

    return dic

def get_Q_FI0(fit,f_hi=9.,f_lo=.9,v=4./1.78,):
    """
    Get the values of Q and FI0 for an event using the fitting parameters
    """
    
    if fit[0] > 0:
        # implies a negative Q, impossible. raise an error
        raise ValueError('A positive gradient implies a negative Q')
    omega_hi=2*np.pi*f_hi
    omega_lo=2*np.pi*f_lo
    
    m=fit[0]
    
    const=(omega_lo-omega_hi)*np.log10(np.e)
    
    return (1/(2*m*v))*const,fit[1]

def srphaWrite(d,fname='./out.srpha'):
        origintime=d[0]
        year=int(origintime.strftime('%Y'))
        julday=int(origintime.strftime('%j'))
        hour=int(origintime.strftime('%H'))
        minute=int(origintime.strftime('%M'))
        sec=float(origintime.strftime('%S.%f')) 
        lon=d[2]
        lat=d[1]
        dep=d[3]
        erh=0.
        x,y=lonlat2xy(lon,lat)
        if x < 0 or y < 0:
            raise ValueError('NOOOO',lon,lat,x,y)
        with open(fname,'w') as fid:
            fid.write('{0:4d}{1:4d}{2:3d}{3:3d}{4:9.4f}{5:10.5f} {6:10.5f}{7:9.4f}                 {8:10.3f}\n'.format(year,julday,hour,minute,sec,x,y,dep,erh))
       
        picks=d[-2]
        for phase in ['P','S']:
            for station in picks[phase]:
                picktime=picks[phase][station][0]
                error=picks[phase][station][1]
                pyear=int(picktime.strftime('%Y'))
                pjulday=int(picktime.strftime('%j'))
                phour=int(picktime.strftime('%H'))
                pminute=int(picktime.strftime('%M'))
                psec=float(picktime.strftime('%S.%f'))
                with open(fname,'a') as fid:
                    fid.write('{0:6}{1:4d}{2:4d}{3:3d}{4:3d}{5:8.4f} {6:1}        {7:8.4f}\n'.format(station,pyear,pjulday,phour,pminute,psec,phase,error))
        with open(fname,'a') as fid:
            fid.write('\n')
        return fname

def get_raypaths(fname,uid):
    """Use subprocess to run raytracing with Punch"""
    with open('./a_script.sh','w') as fid:
        fid.write('''
cd /noc/soes/gg/riftvolc/tim/RiftVolc/bora/tomography/raytracing_1d/velest_1d/Initial
cp /noc/soes/gg/riftvolc/tim/RiftVolc/frequency_content/{0:s} ./\n'.format(fname))
telrayderv bora.spec > /dev/null
rm ./rays/*
mv *.ray ./rays
cp -r rays /noc/soes/gg/riftvolc/tim/RiftVolc/frequency_content/rays/{0:s}\n'.format(uid))'''.format())

    subprocess.call(['bash','./a_script.sh'])
    os.remove('./a_script.sh')
    return True
#################################################################################################
#################################################################################################
#################################################################################################

###############################
##### INPUT PARAMS ############
###############################


def run():
	dist_limit=99999
	freq_lo=(.6,1.2)
	freq_hi=(6,12)
	#freq_lo=(1,2)
	#freq_hi=(10,20)
	#Qdepth=np.linspace(-3,10,14)
	#Qdepth=np.asarray([-3,0,.5,1,1.5,2.,2.5,3.,3.5,4.,4.5,5,6,7,8,9,10])
	Qdepth=np.asarray([-3,-2,-1,0,1,2,3,4,5,10])
	###############################

	print('reading event data')
	event_dic=read_events(['./data/low_freq_sorted.txt','./data/eq_catalogue_sorted.txt'])

	print('reading picks dictionary')
	with open('/noc/soes/gg/riftvolc/tim/RiftVolc/magnitudes/bora_all_picked/hypfile.pkl','rb') as fid:
		pick_dic1=pkl.load(fid)
	with open('/noc/soes/gg/riftvolc/tim/RiftVolc/magnitudes/bora_low_freq/hypfile.pkl','rb') as fid:
		pick_dic2=pkl.load(fid)
	pick_dic={**pick_dic1,**pick_dic2}
	print('calculating FI')
	FI,dist,data,loc_data=calculate_FI('FI.pkl',event_dic,
					   good_stations=['ANOL','ASSE','CHKA','GEBI',
							  'GOND','HURT','ITTE','JIMA',
							  'JIRE','ODAS','OGOL','TULL',
							  'ULAR'],
					   f_lo=freq_lo,f_hi=freq_hi)
	event=data[1]
	stations=data[0]
	hypdist=dist[1]

	#FI=np.asarray([FI[0]])

	ind_events=False
	nobs=len(FI)

	#print(len(FI))
	nev=len(event_dic.keys())
	hdist=dist[1]
	events=data[1]

	# define the Q depth model
	distance_at_depth=np.zeros_like(Qdepth)
	#print('The Q depth model is',Qdepth)

	# get the average velocity in each Q depth layer
	Vdepth=np.zeros_like(Qdepth)
	V=np.loadtxt('/noc/soes/gg/riftvolc/tim/RiftVolc/bora/velest/model/BEST_MODEL.velm')
	d=V[:,0]
	Vs=V[:,2]
	Vs_interp=np.interp(Qdepth,d,Vs)
	Vs_interp_shift=np.hstack((Vs_interp[1:],Vs_interp[-1]))
	Vdepth=(Vs_interp+Vs_interp_shift)/2

	# get some paramters
	n_layers=len(Qdepth)
	n_stations=len(list(set(stations)))
	#print(list(set(stations)))
	n_unknowns=n_stations+n_layers+nev

	# initialise the A matrix
	A=np.zeros((nobs+1,n_unknowns))

	# set the observation vector
	obs=FI

	# middle frequencies
	avg_freq_lo=np.asarray(freq_lo).mean()
	avg_freq_hi=np.asarray(freq_hi).mean()
	OMEGA=(avg_freq_lo-avg_freq_hi)*np.pi*2

	# initialise some other arrays
	unq_ev=np.asarray(list(set(events)))
	unq_ev_pos=np.linspace(0,nev-1,nev,dtype=np.int)
	unq_stat=np.asarray(list(set(stations)))
	unq_stat_pos=np.linspace(0,n_stations-1,n_stations,dtype=np.int)
	unq_layer_pos=np.linspace(0,n_layers-1,n_layers,dtype=np.int)

	print('Populating A matrix')
	from time import time
	for i in range(nobs):
	#        print(i,'of',nobs)
		t0=time()
		uid=events[i]
		fi=FI[i]
		station=stations[i]
		edic=event_dic[uid]

	#        print(edic)
	#        print(pick_dic[uid])
		t1=time()

		t2=time()
		## calculate the raypaths ##
		if not os.path.exists('./rays/'+uid):
		    srpha_file=srphaWrite(pick_dic[uid])
		    get_raypaths(srpha_file,uid)

		rayf='./rays/{0:s}/{1:s}_event1_S.ray'.format(uid,station[:-1])
		if not os.path.exists(rayf):
		    rayf='./rays/{0:s}/{1:s}_event1_P.ray'.format(uid,station[:-1])
		    
		raypath=np.loadtxt(rayf)
		raypath[:,2]*=-1 # change to depth
		x=raypath[:,0]
		y=raypath[:,1]
		z=raypath[:,2]
		max_z=np.max(z)
	#        print(raypath)
	 
		t3=time()
		distance=np.zeros_like(Qdepth)
		dist=np.sqrt(np.power(x[1:]-x[:-1],2)+
			     np.power(y[1:]-y[:-1],2)+
			     np.power(z[1:]-z[:-1],2))

		n_ray=len(raypath)
		k=1
		while k < n_layers:
		    distance[k-1]+=np.sum(dist[(z[1:] >Qdepth[k-1]) & (z[1:] < Qdepth[k])])
		    distance_at_depth[k-1]+=np.sum(dist[(z[1:] >Qdepth[k-1]) & (z[1:] < Qdepth[k])])
		    if Qdepth[k-1] > max_z:
			break
		    k+=1
	#        for D,DI,V in zip(Qdepth,distance,Vdepth):
	#            print('{0:4.1f} {1:3.1f} {2:3.1f}'.format(D,DI,V))
	#        print(distance.sum(),hdist[i])
		
		t4=time()
		# populate the A array
		# first the FI0 term
		pos=unq_ev_pos[unq_ev==uid]
		A[i,pos]=1
		# now the station correction
		pos=unq_stat_pos[unq_stat==station]+nev
		A[i,pos]=1
		# now the layer terms
		A[i,nev+n_stations:nev+n_stations+n_layers]=(distance/Vdepth)*OMEGA*np.log10(np.e)

		i+=1
		t5=time()
		#print('total:',t5-t0,'rays:',t3-t2,'distance at depth:',t4-t3,'srpha:',t2-t1)
	# add a requirement that the station corrections sum to 0
	obs=np.hstack((obs,0))
	A[-1,nev:nev+n_stations]=1
	print(A.shape,obs.shape)

	#print(obs)
	#print(A[:,0])

	#x,res,rank,s=np.linalg.lstsq(A,obs)

	# set up bounded Q
	#from scipy.optimize import lsq_linear
	#lower_bnd=np.ones(n_unknowns)
	#upper_bnd=np.ones_like(lower_bnd)
	#lower_bnd[:nev+n_stations]*=-np.inf
	#lower_bnd[nev+n_stations:]=0
	#upper_bnd*=np.inf

	#x=lsq_linear(A,obs,(lower_bnd,upper_bnd))

	# assemble a regularisation matrix with only the Q portions of the matrix filled
	reg=np.identity(n_unknowns)*2+np.diag(-np.ones(n_unknowns-1),k=1)+np.diag(-np.ones(n_unknowns-1),k=-1)
	reg[:nev+n_stations,:]=reg[:,:nev+n_stations]=0

	print('Solving least-squares')
	fig=plt.figure(figsize=(15,6))
	ax_Qdepth=fig.add_subplot(121)
	ax_elbow=fig.add_subplot(143)
	costs=[]
	smths=[]
	for damp in [1e4]: #[0,1,1e2,1e4,1e6]:
	    res=get_model(A,obs,lamb=damp,regularization=reg,bounds=True,verbose=1)
	    cost=res.cost
	    costs.append(cost)
	    x=res.x
	    #print(x)
	    
	    FI0=x[:nev]
	    stcorr=x[nev:nev+n_stations]
	    Qi=x[nev+n_stations:]
	    Q=1./Qi
	    
	    smth=calculate_smoothing(Q,Qdepth)
	    smths.append(smth)
	  #  ant_loc.append([cost,smth])
	 #   ant_txt.append('{0:d}'.format(damp))

	#    plt.figure()
	#    plt.hist(FI0,bins=20)

	#    plt.figure()
	    Qd=np.asarray([[Qdepth[i],Qdepth[i+1]] for i in range(len(Qdepth)-1)]).flatten()
	    Qp=np.asarray([[Q[i],Q[i]] for i in range(len(Q)-1)]).flatten()
	    ax_Qdepth.plot(Qp,Qd,'k-',label='{0:1.0e}'.format(damp))

	#    print(stcorr)

	#ax_elbow.plot(costs,smths,'o-') #,label='{0:3.1e}'.format(damp))
	#ax_elbow.annotate(ant_txt,ant_loc)

	depths=loc_data[-1]
	dep_events=[depths[event==uid][0] for uid in event_dic.keys()]
	ax_elbow.hist(dep_events,bins=Qdepth,orientation='horizontal',fc='gray')
	ax_elbow.invert_yaxis()
	ax_elbow.set_xlabel('Number of events in bin')
	ax_elbow.set_yticklabels([])
	ax_elbow.set_xlim(0,159)

	#ax_Qdepth.legend(loc=0)
	#ax_elbow.legend(loc=0)
	ax_Qdepth.set_xlim(0,499)
	ax_Qdepth.invert_yaxis()
	ax_Qdepth.set_ylabel('Depth bsl / km')
	ax_Qdepth.set_xlabel('Q')

	ax_data=fig.add_subplot(144)
	dp=np.asarray([[distance_at_depth[i],distance_at_depth[i]] for i in range(len(distance_at_depth)-1)]).flatten()
	ax_data.plot(dp/1000,Qd,'k-')
	ax_data.set_xlabel('distance in layer / 1000*km')
	ax_data.invert_yaxis()
	ax_data.set_yticklabels([])
	fig.subplots_adjust(wspace=.001)
	fig.savefig('./FI_corr_Qinversion.pdf')

	fig=plt.figure()
	plt.hist(FI0,bins=50,fc='gray')
	plt.axvline(0,ls='--',c='k',lw=2)
	plt.xlabel('FI$_0$')
	plt.ylabel('Count')
	plt.savefig('FI0_hist_Qinversion.pdf')

	# write out a file with the low frequency uids


	try:
	    os.rem('./low_frequency_events.uid')
	except OSError as e:
	    pass
	finally:
	    print('this will always be executed')


	for uid in sorted(unq_ev[FI0<0]):
	    print(uid)
	    with open('./low_frequency_events.uid','w') as fid:
		fid.write(uid+'\n')
	      
	# write out a dictionary of the station corrections  
	stcorr=dict(zip(unq_stat,stcorr))
	with open('./stcorr_Qinversion.pkl','wb') as fid:
	    pkl.dump(stcorr,fid)

	# write out a dictiomnary of FI0
	FI0=dict(zip(unq_ev,FI0))
	with open('./FI0_Qinversion.pkl','wb') as fid:
	    pkl.dump(stcorr,fid)

	# write the Q model to a file
	with open('Q_Qinversion.txt','w') as fid:
	    for Q,depth in zip(Q,Qdepth):
		fid.write('{1:5.1f} {0:4.0f}\n'.format(Q,depth))
	plt.show()



	# print(x[nev:nev+n_stations])
	# print(x[nev+n_stations:])


if __name__ == '__main__':
	run()
